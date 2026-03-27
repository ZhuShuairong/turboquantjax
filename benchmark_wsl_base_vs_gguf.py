from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPORT_PATH = Path("/mnt/c/Users/zshua/Downloads/TQ-Experimentation/benchmark_qwen35_turboquant_rotorquant.md")

MODEL_PAIRS = [
    {
        "name": "Qwen3.5-4B",
        "base": Path("/mnt/c/models/base/Qwen3.5-4B-Base"),
        "gguf": Path("/mnt/c/models/gguf/Qwen3.5-4B-IQ4_XS.gguf"),
    },
    {
        "name": "Qwen3.5-9B",
        "base": Path("/mnt/c/models/base/Qwen3.5-9B-Base"),
        "gguf": Path("/mnt/c/models/gguf/Qwen3.5-9B-IQ4_XS.gguf"),
    },
]

SECTION_MARKER = "## WSL Fair Base vs GGUF Inference Comparison"


class GpuMemTracker:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.peak_mb = 0.0

    @staticmethod
    def _query_used_mb() -> float:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return 0.0
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            return 0.0
        try:
            return float(lines[0])
        except ValueError:
            return 0.0

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.peak_mb = max(self.peak_mb, self._query_used_mb())
            time.sleep(self.interval_s)

    def __enter__(self):
        self.peak_mb = self._query_used_mb()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join(timeout=2)


@dataclass
class BenchRow:
    family: str
    model: str
    context_tokens: int
    load_s: float
    prefill_tps: float
    decode_tps: float
    effective_tps: float
    raw_kv_mb: float | None
    context_rss_delta_mb: float
    peak_gpu_mb: float
    peak_rss_mb: float


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{digits}f}"


def _build_prompt_text(tokenizer: AutoTokenizer, target_tokens: int) -> str:
    filler = (
        "This benchmark message tests context scaling, generation throughput, and memory behavior "
        "for fair inference comparison under identical workload settings. "
    )
    text = ""
    while True:
        text += filler
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count >= target_tokens:
            return text


def _build_prompt_tokens_gguf(llm: Any, target_tokens: int) -> list[int]:
    filler = (
        "This benchmark message tests context scaling, generation throughput, and memory behavior "
        "for fair inference comparison under identical workload settings. "
    )
    text = ""
    toks: list[int] = []
    while len(toks) < target_tokens:
        text += filler
        toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    return toks[:target_tokens]


def _extend_ld_library_path(paths: list[Path]) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(":") if p]
    existing_set = set(existing_parts)

    new_parts = []
    for path in paths:
        p = str(path)
        if path.exists() and p not in existing_set:
            new_parts.append(p)
            existing_set.add(p)

    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts + existing_parts)


def _import_llama_cpp() -> Any:
    candidates: list[Path] = []
    for package_root in {Path(p) for p in sys.path if "site-packages" in p}:
        candidates.append(package_root / "nvidia" / "cuda_runtime" / "lib")
        candidates.append(package_root / "nvidia" / "cublas" / "lib")
    candidates.append(Path(sys.prefix) / "lib")
    _extend_ld_library_path(candidates)

    import llama_cpp  # type: ignore

    return llama_cpp


def _cache_bytes_total(cache: Any) -> int:
    def tensor_bytes(value: Any) -> int:
        if torch.is_tensor(value):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(v) for v in value)
        return 0

    if hasattr(cache, "recurrent_states") and hasattr(cache, "conv_states"):
        return tensor_bytes(list(cache.recurrent_states)) + tensor_bytes(list(cache.conv_states))
    if hasattr(cache, "layers"):
        keys = []
        values = []
        for layer in cache.layers:
            keys.append(layer.keys)
            values.append(layer.values)
        return tensor_bytes(keys) + tensor_bytes(values)
    return tensor_bytes(cache)


def benchmark_base_model(model_name: str, model_dir: Path, contexts: list[int], new_tokens: int) -> list[BenchRow]:
    process = psutil.Process()

    rss_before_load = process.memory_info().rss / (1024 * 1024)
    with GpuMemTracker() as tracker:
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.float16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_s = time.perf_counter() - load_start
        peak_gpu_load = tracker.peak_mb

    # Warm-up to avoid first-pass effects in measured contexts.
    warm_len = min(64, min(contexts))
    warm_text = _build_prompt_text(tokenizer, warm_len)
    warm_inputs = tokenizer(
        warm_text,
        return_tensors="pt",
        truncation=True,
        max_length=warm_len,
    ).to("cuda")
    with torch.inference_mode():
        warm_out = model(**warm_inputs, use_cache=True)
        warm_past = warm_out.past_key_values
        warm_next = warm_inputs["input_ids"][:, -1:]
        for _ in range(min(4, new_tokens)):
            warm_out = model(input_ids=warm_next, past_key_values=warm_past, use_cache=True)
            warm_past = warm_out.past_key_values
            warm_next = warm_out.logits[:, -1].argmax(dim=-1, keepdim=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    rows: list[BenchRow] = []
    for context_tokens in contexts:
        prompt_text = _build_prompt_text(tokenizer, context_tokens)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=context_tokens,
        ).to("cuda")

        rss_before_ctx = process.memory_info().rss / (1024 * 1024)
        with GpuMemTracker() as tracker:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                out = model(**inputs, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_s = time.perf_counter() - prefill_start

            past = out.past_key_values
            raw_kv_mb = _cache_bytes_total(past) / (1024 * 1024)

            next_token = inputs["input_ids"][:, -1:]
            decode_start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(new_tokens):
                    out = model(input_ids=next_token, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_s = time.perf_counter() - decode_start

            peak_gpu_ctx = tracker.peak_mb

        rss_after_ctx = process.memory_info().rss / (1024 * 1024)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        rows.append(
            BenchRow(
                family="base-fp16",
                model=model_name,
                context_tokens=prompt_tokens,
                load_s=load_s,
                prefill_tps=prompt_tokens / prefill_s,
                decode_tps=new_tokens / decode_s,
                effective_tps=(prompt_tokens + new_tokens) / (prefill_s + decode_s),
                raw_kv_mb=raw_kv_mb,
                context_rss_delta_mb=max(0.0, rss_after_ctx - rss_before_ctx),
                peak_gpu_mb=max(peak_gpu_load, peak_gpu_ctx),
                peak_rss_mb=max(rss_before_load, rss_after_ctx),
            )
        )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    return rows


def benchmark_gguf_model(model_name: str, model_path: Path, contexts: list[int], new_tokens: int, n_gpu_layers: int) -> list[BenchRow]:
    process = psutil.Process()
    llama_cpp = _import_llama_cpp()
    Llama = llama_cpp.Llama

    rss_before_load = process.memory_info().rss / (1024 * 1024)

    candidate_layers: list[int]
    if n_gpu_layers == -1:
        candidate_layers = [-1, 80, 60, 48, 40, 32, 24, 16, 8, 0]
    else:
        candidate_layers = [n_gpu_layers] + [v for v in [80, 60, 48, 40, 32, 24, 16, 8, 0] if v <= n_gpu_layers]
    # Keep order stable while removing duplicates.
    candidate_layers = list(dict.fromkeys(candidate_layers))

    llm = None
    selected_layers = n_gpu_layers
    last_error: Exception | None = None

    with GpuMemTracker() as tracker:
        load_start = time.perf_counter()
        for layer_setting in candidate_layers:
            try:
                llm = Llama(
                    model_path=str(model_path),
                    n_ctx=max(contexts) + new_tokens + 64,
                    n_threads=min(8, os.cpu_count() or 8),
                    n_threads_batch=min(8, os.cpu_count() or 8),
                    n_batch=512,
                    n_gpu_layers=layer_setting,
                    verbose=False,
                )
                selected_layers = layer_setting
                break
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                gc.collect()
                time.sleep(0.3)

        if llm is None:
            raise RuntimeError(
                f"Failed to load GGUF model {model_path} with candidate n_gpu_layers={candidate_layers}"
            ) from last_error

        load_s = time.perf_counter() - load_start
        peak_gpu_load = tracker.peak_mb

    # Warm-up to avoid first-pass effects in measured contexts.
    warm_len = min(64, min(contexts))
    warm_tokens = _build_prompt_tokens_gguf(llm, warm_len)
    llm.reset()
    llm.eval(warm_tokens)
    for _ in range(min(4, new_tokens)):
        token = int(llm.sample(top_k=1, temp=0.0, repeat_penalty=1.0))
        llm.eval([token])

    rows: list[BenchRow] = []
    for context_tokens in contexts:
        prompt_tokens = _build_prompt_tokens_gguf(llm, context_tokens)

        rss_before_ctx = process.memory_info().rss / (1024 * 1024)
        with GpuMemTracker() as tracker:
            llm.reset()
            prefill_start = time.perf_counter()
            llm.eval(prompt_tokens)
            prefill_s = time.perf_counter() - prefill_start

            decode_start = time.perf_counter()
            for _ in range(new_tokens):
                token = int(llm.sample(top_k=1, temp=0.0, repeat_penalty=1.0))
                llm.eval([token])
            decode_s = time.perf_counter() - decode_start

            peak_gpu_ctx = tracker.peak_mb

        rss_after_ctx = process.memory_info().rss / (1024 * 1024)

        rows.append(
            BenchRow(
                family=f"gguf-iq4_xs(layers={selected_layers})",
                model=model_name,
                context_tokens=len(prompt_tokens),
                load_s=load_s,
                prefill_tps=len(prompt_tokens) / prefill_s,
                decode_tps=new_tokens / decode_s,
                effective_tps=(len(prompt_tokens) + new_tokens) / (prefill_s + decode_s),
                raw_kv_mb=None,
                context_rss_delta_mb=max(0.0, rss_after_ctx - rss_before_ctx),
                peak_gpu_mb=max(peak_gpu_load, peak_gpu_ctx),
                peak_rss_mb=max(rss_before_load, rss_after_ctx),
            )
        )

    if hasattr(llm, "close"):
        llm.close()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    time.sleep(0.5)
    return rows


def _render_section(rows: list[BenchRow], contexts: list[int], new_tokens: int, n_gpu_layers: int) -> str:
    lines: list[str] = []
    lines.append(SECTION_MARKER)
    lines.append("")
    lines.append("- Environment: WSL Ubuntu, same conda runtime for both base and GGUF benchmarks")
    lines.append(f"- Decode tokens: {new_tokens}")
    lines.append(f"- Context tokens tested: {', '.join(str(c) for c in contexts)}")
    lines.append(f"- GGUF n_gpu_layers: {n_gpu_layers}")
    lines.append("")

    lines.append("| Family | Model | Context tokens | Load (s) | Prefill tok/s | Decode tok/s | Effective tok/s | Raw KV cache (MB) | Context RSS delta (MB) | Peak GPU (MB) | Peak RSS (MB) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {family} | {model} | {ctx} | {load} | {prefill} | {decode} | {effective} | {raw_kv} | {rss_delta} | {gpu} | {rss_peak} |".format(
                family=row.family,
                model=row.model,
                ctx=row.context_tokens,
                load=_fmt(row.load_s, 2),
                prefill=_fmt(row.prefill_tps, 1),
                decode=_fmt(row.decode_tps, 1),
                effective=_fmt(row.effective_tps, 1),
                raw_kv=_fmt(row.raw_kv_mb, 1),
                rss_delta=_fmt(row.context_rss_delta_mb, 1),
                gpu=_fmt(row.peak_gpu_mb, 1),
                rss_peak=_fmt(row.peak_rss_mb, 1),
            )
        )

    lines.append("")
    lines.append("### Context Scaling Summary")
    lines.append("")
    lines.append("| Model | Family | Prefill tok/s @ min context | Prefill tok/s @ max context | Prefill scaling | Decode tok/s @ min context | Decode tok/s @ max context |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")

    by_key: dict[tuple[str, str], list[BenchRow]] = {}
    for r in rows:
        by_key.setdefault((r.model, r.family), []).append(r)

    for (model, family), vals in sorted(by_key.items()):
        vals = sorted(vals, key=lambda x: x.context_tokens)
        first = vals[0]
        last = vals[-1]
        scaling = (last.prefill_tps / first.prefill_tps) if first.prefill_tps > 0 else 0.0
        lines.append(
            "| {model} | {family} | {p0} | {p1} | {scale}x | {d0} | {d1} |".format(
                model=model,
                family=family,
                p0=_fmt(first.prefill_tps, 1),
                p1=_fmt(last.prefill_tps, 1),
                scale=_fmt(scaling, 2),
                d0=_fmt(first.decode_tps, 1),
                d1=_fmt(last.decode_tps, 1),
            )
        )

    lines.append("")
    lines.append("### Readout")
    lines.append("")
    lines.append("- Base and GGUF were benchmarked in WSL under one runtime stack for direct comparison.")
    lines.append("- Prefill scaling captures context-length impact; decode tok/s captures generation throughput under the same decode length.")
    lines.append("- Raw KV cache is directly reported for base models; GGUF rows use context RSS delta as a practical memory growth proxy.")
    lines.append("")
    return "\n".join(lines)


def _replace_section(report_path: Path, marker: str, section: str) -> None:
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if marker in text:
        before, _, _ = text.partition(marker)
        updated = before.rstrip() + "\n\n" + section.strip() + "\n"
    else:
        updated = text.rstrip() + "\n\n" + section.strip() + "\n" if text.strip() else section.strip() + "\n"
    report_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair WSL benchmark for base vs GGUF inference context and throughput")
    parser.add_argument("--report-path", default=str(REPORT_PATH), help="Parent markdown report path")
    parser.add_argument("--contexts", nargs="*", type=int, default=[256, 512, 1024], help="Prompt context token lengths")
    parser.add_argument("--decode-tokens", type=int, default=24, help="Decode token length")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="llama.cpp n_gpu_layers")
    parser.add_argument("--models", nargs="*", default=[p["name"] for p in MODEL_PAIRS], help="Model pair names to run")
    args = parser.parse_args()

    report_path = Path(args.report_path)
    contexts = sorted(set(int(c) for c in args.contexts if c > 0))

    selected_pairs = [p for p in MODEL_PAIRS if p["name"] in set(args.models)]
    if not selected_pairs:
        raise ValueError("No valid model pair selected")

    rows: list[BenchRow] = []
    for pair in selected_pairs:
        if not pair["base"].exists():
            raise FileNotFoundError(pair["base"])
        if not pair["gguf"].exists():
            raise FileNotFoundError(pair["gguf"])

    for pair in selected_pairs:
        name = pair["name"]
        gguf_path = pair["gguf"]
        print(f"Running GGUF benchmark for {name}...", flush=True)
        rows.extend(benchmark_gguf_model(name, gguf_path, contexts, args.decode_tokens, args.n_gpu_layers))

    for pair in selected_pairs:
        name = pair["name"]
        base_dir = pair["base"]
        print(f"Running base benchmark for {name}...", flush=True)
        rows.extend(benchmark_base_model(name, base_dir, contexts, args.decode_tokens))

    section = _render_section(rows, contexts, args.decode_tokens, args.n_gpu_layers)
    _replace_section(report_path, SECTION_MARKER, section)
    print(f"Updated report at {report_path}")


if __name__ == "__main__":
    main()

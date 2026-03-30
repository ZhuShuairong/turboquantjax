from __future__ import annotations

import argparse
import os
import ctypes
import gc
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Avoid aggressive GPU preallocation that can starve desktop rendering on low-VRAM systems.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_jax import JAXTurboQuantKVCache  # noqa: E402

REPORT_PATH = Path("/mnt/c/Users/zshua/Downloads/TQ-Experimentation/benchmark_qwen35_turboquant_rotorquant.md")
SECTION_MARKER = "## TurboQuant vs llama.cpp KV Cache Benchmark"

CACHE_TYPE_ALIASES = {
    "f16": "GGML_TYPE_F16",
    "q8_0": "GGML_TYPE_Q8_0",
    "q4_0": "GGML_TYPE_Q4_0",
    "q4_1": "GGML_TYPE_Q4_1",
    "q5_0": "GGML_TYPE_Q5_0",
    "q5_1": "GGML_TYPE_Q5_1",
}

# Effective bits include block overhead for common ggml quant formats.
EFFECTIVE_BITS = {
    "f16": 16.0,
    "q8_0": 8.5,
    "q4_0": 4.5,
    "q4_1": 5.0,
    "q5_0": 5.5,
    "q5_1": 6.0,
}


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
class LlamaKvRow:
    backend: str
    model_name: str
    cache_type: str
    context_tokens: int
    status: str
    selected_gpu_layers: int | None
    load_s: float | None
    prefill_tps: float | None
    decode_tps: float | None
    effective_tps: float | None
    est_kv_mb: float | None
    est_max_ctx_8gb: int | None
    peak_gpu_mb: float | None
    rss_delta_mb: float | None
    error: str | None


@dataclass
class TurboKvRow:
    backend: str
    model_name: str
    policy: str
    bits: int
    context_tokens: int
    append_s: float
    score_tps: float
    stored_kv_mb: float
    runtime_kv_mb: float
    fp16_kv_mb: float
    compression_ratio: float
    runtime_compression_ratio: float
    est_max_ctx_8gb: int


@dataclass
class ModelRunInfo:
    model_name: str
    model_path: Path
    dims: dict[str, int]


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{digits}f}"


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _block_tree(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _replace_section(report_path: Path, marker: str, section: str) -> None:
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if marker in text:
        before, _, _ = text.partition(marker)
        updated = before.rstrip() + "\n\n" + section.strip() + "\n"
    else:
        updated = text.rstrip() + "\n\n" + section.strip() + "\n" if text.strip() else section.strip() + "\n"
    report_path.write_text(updated, encoding="utf-8")


def _infer_model_name_from_path(model_path: Path) -> str:
    return model_path.stem


def _cooldown(seconds: float) -> None:
    if seconds <= 0:
        return
    gc.collect()
    time.sleep(seconds)


def _discover_gguf_models(
    gguf_root: Path,
    include_substr: str,
    exclude_substr: str,
) -> list[Path]:
    if not gguf_root.exists():
        raise FileNotFoundError(gguf_root)

    include_s = include_substr.lower().strip()
    exclude_s = exclude_substr.lower().strip()

    out: list[Path] = []
    for path in sorted(gguf_root.glob("*.gguf")):
        name = path.name.lower()
        if include_s and include_s not in name:
            continue
        if exclude_s and exclude_s in name:
            continue
        out.append(path)
    return out


def _preload_llama_cuda_libs() -> None:
    package_roots = {Path(p) for p in sys.path if "site-packages" in p}
    candidates = []
    for root in package_roots:
        candidates.append(root / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12")
        candidates.append(root / "nvidia" / "cublas" / "lib" / "libcublas.so.12")
        candidates.append(root / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12")

    for lib in candidates:
        if lib.exists():
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def _import_llama_cpp() -> Any:
    _preload_llama_cuda_libs()
    import llama_cpp  # type: ignore

    return llama_cpp


def _candidate_layers(requested: int) -> list[int]:
    base = [-1, 80, 60, 48, 40, 32, 24, 16, 8, 0]
    if requested == -1:
        return base
    return list(dict.fromkeys([requested] + [x for x in base if x <= requested]))


def _build_prompt_tokens(llm: Any, target_tokens: int) -> list[int]:
    filler = (
        "Long-context benchmark text for KV cache scaling. "
        "This payload is repeated to fill prompt context while preserving deterministic behavior. "
    )
    text = ""
    toks: list[int] = []
    while len(toks) < target_tokens:
        text += filler
        toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    return toks[:target_tokens]


def _resolve_cache_type_enum(llama_cpp: Any, cache_type: str) -> int:
    key = cache_type.lower().strip()
    if key not in CACHE_TYPE_ALIASES:
        raise ValueError(f"Unsupported cache type: {cache_type}")
    attr = CACHE_TYPE_ALIASES[key]
    if not hasattr(llama_cpp, attr):
        raise ValueError(f"llama_cpp missing enum constant: {attr}")
    return int(getattr(llama_cpp, attr))


def _infer_arch_dims(
    model_path: Path,
    llama_threads: int,
    llama_threads_batch: int,
    llama_n_batch: int,
) -> dict[str, int]:
    llama_cpp = _import_llama_cpp()
    Llama = llama_cpp.Llama

    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,
        n_gpu_layers=0,
        n_threads=max(1, int(llama_threads)),
        n_threads_batch=max(1, int(llama_threads_batch)),
        n_batch=max(32, int(llama_n_batch)),
        verbose=False,
    )
    try:
        md = dict(llm.metadata)
        arch = str(md.get("general.architecture", "qwen35"))

        block_count = _to_int(md.get(f"{arch}.block_count"), 0)
        kv_heads = _to_int(md.get(f"{arch}.attention.head_count_kv"), 0)
        attn_heads = _to_int(md.get(f"{arch}.attention.head_count"), 0)
        key_length = _to_int(md.get(f"{arch}.attention.key_length"), 0)
        value_length = _to_int(md.get(f"{arch}.attention.value_length"), 0)
        embedding_length = _to_int(md.get(f"{arch}.embedding_length"), 0)
        context_length = _to_int(md.get(f"{arch}.context_length"), 0)

        if block_count <= 0:
            block_count = _to_int(md.get("llama.block_count"), 0)
        if kv_heads <= 0:
            kv_heads = _to_int(md.get("llama.attention.head_count_kv"), 0)
        if attn_heads <= 0:
            attn_heads = _to_int(md.get("llama.attention.head_count"), 0)
        if key_length <= 0:
            key_length = _to_int(md.get("llama.attention.key_length"), 0)
        if value_length <= 0:
            value_length = _to_int(md.get("llama.attention.value_length"), 0)
        if embedding_length <= 0:
            embedding_length = _to_int(md.get("llama.embedding_length"), 0)
        if context_length <= 0:
            context_length = _to_int(md.get("llama.context_length"), 0)

        if key_length <= 0 and embedding_length > 0 and attn_heads > 0:
            key_length = embedding_length // attn_heads
        if value_length <= 0:
            value_length = key_length
        if kv_heads <= 0:
            kv_heads = max(1, attn_heads)

        return {
            "layers": max(1, block_count),
            "kv_heads": max(1, kv_heads),
            "head_dim_k": max(1, key_length),
            "head_dim_v": max(1, value_length),
            "train_ctx": max(1, context_length),
            "attn_heads": max(1, attn_heads),
        }
    finally:
        if hasattr(llm, "close"):
            llm.close()


def _llama_kv_bytes(layers: int, kv_heads: int, d_k: int, d_v: int, context_tokens: int, bits_per_elem: float) -> float:
    bits_per_token = float(layers) * float(kv_heads) * (float(d_k) + float(d_v)) * float(bits_per_elem)
    return float(context_tokens) * bits_per_token / 8.0


def _estimate_max_ctx_from_bytes_per_token(bytes_per_token: float, vram_budget_mb: float) -> int:
    if bytes_per_token <= 0:
        return 0
    budget_bytes = float(vram_budget_mb) * 1024.0 * 1024.0
    return int(budget_bytes // bytes_per_token)


def _benchmark_llama_cpp(
    model_name: str,
    model_path: Path,
    contexts: list[int],
    decode_tokens: int,
    n_gpu_layers: int,
    cache_types: list[str],
    dims: dict[str, int],
    vram_budget_mb: float,
    llama_threads: int,
    llama_threads_batch: int,
    llama_n_batch: int,
    run_cooldown_s: float,
) -> list[LlamaKvRow]:
    process = psutil.Process()
    llama_cpp = _import_llama_cpp()
    Llama = llama_cpp.Llama
    rows: list[LlamaKvRow] = []

    for cache_type in cache_types:
        effective_bits = EFFECTIVE_BITS.get(cache_type.lower().strip())
        if effective_bits is None:
            rows.extend(
                LlamaKvRow(
                    backend="llama.cpp",
                    model_name=model_name,
                    cache_type=cache_type,
                    context_tokens=ctx,
                    status="skip",
                    selected_gpu_layers=None,
                    load_s=None,
                    prefill_tps=None,
                    decode_tps=None,
                    effective_tps=None,
                    est_kv_mb=None,
                    est_max_ctx_8gb=None,
                    peak_gpu_mb=None,
                    rss_delta_mb=None,
                    error=f"No effective-bit mapping for cache type {cache_type}",
                )
                for ctx in contexts
            )
            continue

        try:
            type_enum = _resolve_cache_type_enum(llama_cpp, cache_type)
        except Exception as exc:
            rows.extend(
                LlamaKvRow(
                    backend="llama.cpp",
                    model_name=model_name,
                    cache_type=cache_type,
                    context_tokens=ctx,
                    status="error",
                    selected_gpu_layers=None,
                    load_s=None,
                    prefill_tps=None,
                    decode_tps=None,
                    effective_tps=None,
                    est_kv_mb=None,
                    est_max_ctx_8gb=None,
                    peak_gpu_mb=None,
                    rss_delta_mb=None,
                    error=str(exc),
                )
                for ctx in contexts
            )
            continue

        for context_tokens in contexts:
            llm = None
            selected_layers: int | None = None
            load_s: float | None = None
            peak_gpu_load: float | None = None
            last_error: Exception | None = None

            target_ctx = int(context_tokens + decode_tokens + 64)
            with GpuMemTracker() as tracker:
                load_start = time.perf_counter()
                for candidate in _candidate_layers(n_gpu_layers):
                    try:
                        llm = Llama(
                            model_path=str(model_path),
                            n_ctx=target_ctx,
                            n_threads=max(1, int(llama_threads)),
                            n_threads_batch=max(1, int(llama_threads_batch)),
                            n_batch=max(32, int(llama_n_batch)),
                            n_gpu_layers=candidate,
                            type_k=type_enum,
                            type_v=type_enum,
                            verbose=False,
                        )
                        selected_layers = candidate
                        break
                    except Exception as exc:  # noqa: PERF203
                        last_error = exc
                        gc.collect()
                        time.sleep(0.2)

                if llm is not None:
                    load_s = time.perf_counter() - load_start
                    peak_gpu_load = tracker.peak_mb

            kv_bytes = _llama_kv_bytes(
                layers=dims["layers"],
                kv_heads=dims["kv_heads"],
                d_k=dims["head_dim_k"],
                d_v=dims["head_dim_v"],
                context_tokens=context_tokens,
                bits_per_elem=effective_bits,
            )
            kv_mb = kv_bytes / (1024.0 * 1024.0)
            bytes_per_token = kv_bytes / max(float(context_tokens), 1.0)
            max_ctx_8gb = _estimate_max_ctx_from_bytes_per_token(bytes_per_token, vram_budget_mb)

            if llm is None:
                rows.append(
                    LlamaKvRow(
                        backend="llama.cpp",
                        model_name=model_name,
                        cache_type=cache_type,
                        context_tokens=context_tokens,
                        status="load-failed",
                        selected_gpu_layers=selected_layers,
                        load_s=load_s,
                        prefill_tps=None,
                        decode_tps=None,
                        effective_tps=None,
                        est_kv_mb=kv_mb,
                        est_max_ctx_8gb=max_ctx_8gb,
                        peak_gpu_mb=peak_gpu_load,
                        rss_delta_mb=None,
                        error=str(last_error) if last_error is not None else "unknown load error",
                    )
                )
                continue

            try:
                prompt_tokens = _build_prompt_tokens(llm, context_tokens)
                rss_before = process.memory_info().rss / (1024.0 * 1024.0)

                with GpuMemTracker() as tracker:
                    llm.reset()
                    prefill_start = time.perf_counter()
                    llm.eval(prompt_tokens)
                    prefill_s = time.perf_counter() - prefill_start

                    decode_start = time.perf_counter()
                    for _ in range(decode_tokens):
                        token = int(llm.sample(top_k=1, temp=0.0, repeat_penalty=1.0))
                        llm.eval([token])
                    decode_s = time.perf_counter() - decode_start
                    peak_gpu_ctx = tracker.peak_mb

                rss_after = process.memory_info().rss / (1024.0 * 1024.0)
                p_toks = len(prompt_tokens)

                rows.append(
                    LlamaKvRow(
                        backend="llama.cpp",
                        model_name=model_name,
                        cache_type=cache_type,
                        context_tokens=p_toks,
                        status="ok",
                        selected_gpu_layers=selected_layers,
                        load_s=load_s,
                        prefill_tps=(p_toks / prefill_s) if prefill_s > 0 else None,
                        decode_tps=(decode_tokens / decode_s) if decode_s > 0 else None,
                        effective_tps=((p_toks + decode_tokens) / (prefill_s + decode_s))
                        if (prefill_s + decode_s) > 0
                        else None,
                        est_kv_mb=kv_mb,
                        est_max_ctx_8gb=max_ctx_8gb,
                        peak_gpu_mb=max(peak_gpu_load or 0.0, peak_gpu_ctx),
                        rss_delta_mb=max(0.0, rss_after - rss_before),
                        error=None,
                    )
                )
            except Exception as exc:
                rows.append(
                    LlamaKvRow(
                        backend="llama.cpp",
                        model_name=model_name,
                        cache_type=cache_type,
                        context_tokens=context_tokens,
                        status="run-failed",
                        selected_gpu_layers=selected_layers,
                        load_s=load_s,
                        prefill_tps=None,
                        decode_tps=None,
                        effective_tps=None,
                        est_kv_mb=kv_mb,
                        est_max_ctx_8gb=max_ctx_8gb,
                        peak_gpu_mb=peak_gpu_load,
                        rss_delta_mb=None,
                        error=str(exc),
                    )
                )
            finally:
                if hasattr(llm, "close"):
                    llm.close()
                del llm
                gc.collect()
                _cooldown(run_cooldown_s)

    return rows


def _make_kv(key: jax.Array, b: int, h: int, s: int, d_k: int, d_v: int):
    k1, k2 = jax.random.split(key)
    keys = jax.random.normal(k1, (b, h, s, d_k), dtype=jnp.float16)
    values = jax.random.normal(k2, (b, h, s, d_v), dtype=jnp.float16)

    keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
    values = values / (jnp.linalg.norm(values, axis=-1, keepdims=True) + 1e-8)
    return keys, values


def _warmup_turboquant(
    d_k: int,
    d_v: int,
    kv_heads: int,
    bits: int,
    policy: str,
    score_backend: str,
    tile_size: int,
    query_tile_size: int,
) -> None:
    key = jax.random.PRNGKey(123)
    keys, values = _make_kv(key, 1, kv_heads, 64, d_k, d_v)
    cache = JAXTurboQuantKVCache(
        d_key=d_k,
        d_value=d_v,
        bits=bits,
        score_cache_policy=policy,
        score_backend=score_backend,
        tile_size=tile_size,
        query_tile_size=query_tile_size,
    )
    cache.append(keys, values)
    q = keys[:, :, -1:, :]
    s = cache.attention_scores(q)
    _block_tree(s)


def _benchmark_turboquant(
    model_name: str,
    contexts: list[int],
    decode_tokens: int,
    dims: dict[str, int],
    bits_list: list[int],
    policies: list[str],
    score_backend: str,
    tile_size: int,
    query_tile_size: int,
    vram_budget_mb: float,
    device: str,
    run_cooldown_s: float,
) -> list[TurboKvRow]:
    rows: list[TurboKvRow] = []

    devices = jax.devices(device)
    if not devices:
        raise RuntimeError(f"No JAX device found for device type {device}")
    run_device = devices[0]

    for bits in bits_list:
        for policy in policies:
            _warmup_turboquant(
                d_k=dims["head_dim_k"],
                d_v=dims["head_dim_v"],
                kv_heads=dims["kv_heads"],
                bits=bits,
                policy=policy,
                score_backend=score_backend,
                tile_size=tile_size,
                query_tile_size=query_tile_size,
            )

    key = jax.random.PRNGKey(0)
    for context_tokens in contexts:
        key, sub = jax.random.split(key)
        keys, values = _make_kv(
            sub,
            b=1,
            h=dims["kv_heads"],
            s=context_tokens,
            d_k=dims["head_dim_k"],
            d_v=dims["head_dim_v"],
        )
        keys = jax.device_put(keys, run_device)
        values = jax.device_put(values, run_device)
        query = keys[:, :, -1:, :]

        for bits in bits_list:
            for policy in policies:
                cache = JAXTurboQuantKVCache(
                    d_key=dims["head_dim_k"],
                    d_value=dims["head_dim_v"],
                    bits=bits,
                    score_cache_policy=policy,
                    score_backend=score_backend,
                    tile_size=tile_size,
                    query_tile_size=query_tile_size,
                    adaptive_promote_after=2,
                    adaptive_hit_rate_threshold=0.35,
                    adaptive_memory_budget_mb=vram_budget_mb,
                )

                append_start = time.perf_counter()
                cache.append(keys, values)
                append_s = time.perf_counter() - append_start

                score_start = time.perf_counter()
                score = None
                for _ in range(decode_tokens):
                    score = cache.attention_scores(query)
                if score is not None:
                    _block_tree(score)
                score_s = time.perf_counter() - score_start

                mem = cache.memory_usage_bits()
                layer_stored_mb = (mem["total_bits"] / 8.0) / (1024.0 * 1024.0)
                layer_runtime_mb = (mem["runtime_total_bits"] / 8.0) / (1024.0 * 1024.0)
                layer_fp16_mb = (mem["fp16_bits"] / 8.0) / (1024.0 * 1024.0)

                # JAX cache object is one layer; scale by model layer count for end-to-end KV budget.
                stored_mb = layer_stored_mb * float(dims["layers"])
                runtime_mb = layer_runtime_mb * float(dims["layers"])
                fp16_mb = layer_fp16_mb * float(dims["layers"])

                runtime_bytes_per_token = (runtime_mb * 1024.0 * 1024.0) / max(float(context_tokens), 1.0)
                max_ctx_8gb = _estimate_max_ctx_from_bytes_per_token(runtime_bytes_per_token, vram_budget_mb)

                rows.append(
                    TurboKvRow(
                        backend="turboquant-jax",
                        model_name=model_name,
                        policy=policy,
                        bits=bits,
                        context_tokens=context_tokens,
                        append_s=append_s,
                        score_tps=(decode_tokens / score_s) if score_s > 0 else 0.0,
                        stored_kv_mb=stored_mb,
                        runtime_kv_mb=runtime_mb,
                        fp16_kv_mb=fp16_mb,
                        compression_ratio=mem["compression_ratio"],
                        runtime_compression_ratio=mem["runtime_compression_ratio"],
                        est_max_ctx_8gb=max_ctx_8gb,
                    )
                )

        del keys
        del values
        del query
        gc.collect()
        _cooldown(run_cooldown_s)

    return rows


def _render_section(
    model_runs: list[ModelRunInfo],
    contexts: list[int],
    decode_tokens: int,
    vram_budget_mb: float,
    llama_rows: list[LlamaKvRow],
    turbo_rows: list[TurboKvRow],
    score_backend: str,
    run_profile: str,
) -> str:
    lines: list[str] = [SECTION_MARKER, ""]
    lines.append(f"- Models benchmarked: {len(model_runs)}")
    lines.append(f"- Model list: {', '.join(run.model_name for run in model_runs)}")
    lines.append(f"- Context sweep: {', '.join(str(c) for c in contexts)}")
    lines.append(f"- Decode tokens: {decode_tokens}")
    lines.append(f"- VRAM budget target: {vram_budget_mb:.1f} MB (for projected KV-only capacity)")
    lines.append(f"- TurboQuant score backend: {score_backend}")
    lines.append(f"- Run profile: {run_profile}")
    lines.append("")

    lines.append("### Model Metadata")
    lines.append("")
    lines.append("| Model | GGUF path | Layers | KV heads | Key dim | Value dim | Train context |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for run in model_runs:
        dims = run.dims
        lines.append(
            "| {name} | {path} | {layers} | {kvh} | {dk} | {dv} | {train_ctx} |".format(
                name=run.model_name,
                path=run.model_path,
                layers=dims["layers"],
                kvh=dims["kv_heads"],
                dk=dims["head_dim_k"],
                dv=dims["head_dim_v"],
                train_ctx=dims["train_ctx"],
            )
        )

    lines.append("")
    lines.append("### llama.cpp KV Cache Types")
    lines.append("")
    lines.append("| Backend | Model | Cache type | Context | Status | GPU layers | Load (s) | Prefill tok/s | Decode tok/s | Effective tok/s | Est KV (MB) | Est max ctx @ 8GB | Peak GPU (MB) | RSS delta (MB) | Error |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    for r in llama_rows:
        lines.append(
            "| {backend} | {model} | {cache} | {ctx} | {status} | {layers} | {load} | {prefill} | {decode} | {eff} | {kv} | {maxctx} | {gpu} | {rss} | {err} |".format(
                backend=r.backend,
                model=r.model_name,
                cache=r.cache_type,
                ctx=r.context_tokens,
                status=r.status,
                layers=r.selected_gpu_layers if r.selected_gpu_layers is not None else "n/a",
                load=_fmt(r.load_s, 2),
                prefill=_fmt(r.prefill_tps, 1),
                decode=_fmt(r.decode_tps, 1),
                eff=_fmt(r.effective_tps, 1),
                kv=_fmt(r.est_kv_mb, 1),
                maxctx=r.est_max_ctx_8gb if r.est_max_ctx_8gb is not None else "n/a",
                gpu=_fmt(r.peak_gpu_mb, 1),
                rss=_fmt(r.rss_delta_mb, 1),
                err=(r.error or "").replace("|", "/"),
            )
        )

    lines.append("")
    lines.append("### TurboQuant JAX KV Cache")
    lines.append("")
    lines.append("| Backend | Model | Policy | Bits | Context | Append (s) | Score tok/s | Stored KV (MB, all layers) | Runtime KV (MB, all layers) | FP16 KV (MB, all layers) | Stored compression | Runtime compression | Est max ctx @ 8GB |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for r in turbo_rows:
        lines.append(
            "| {backend} | {model} | {policy} | {bits} | {ctx} | {append} | {score} | {stored} | {runtime} | {fp16} | {cr}x | {rcr}x | {maxctx} |".format(
                backend=r.backend,
                model=r.model_name,
                policy=r.policy,
                bits=r.bits,
                ctx=r.context_tokens,
                append=_fmt(r.append_s, 3),
                score=_fmt(r.score_tps, 1),
                stored=_fmt(r.stored_kv_mb, 1),
                runtime=_fmt(r.runtime_kv_mb, 1),
                fp16=_fmt(r.fp16_kv_mb, 1),
                cr=_fmt(r.compression_ratio, 2),
                rcr=_fmt(r.runtime_compression_ratio, 2),
                maxctx=r.est_max_ctx_8gb,
            )
        )

    lines.append("")
    lines.append("### Readout")
    lines.append("")
    max_ctx = max(contexts) if contexts else 0

    for run in model_runs:
        model = run.model_name
        llama_ok = [r for r in llama_rows if r.model_name == model and r.status == "ok" and r.est_kv_mb is not None]
        turbo_ok = [r for r in turbo_rows if r.model_name == model]

        if llama_ok and turbo_ok and max_ctx > 0:
            llama_at_max = [r for r in llama_ok if r.context_tokens == max_ctx]
            turbo_at_max = [r for r in turbo_ok if r.context_tokens == max_ctx]

            best_llama = min(
                llama_at_max if llama_at_max else llama_ok,
                key=lambda x: x.est_kv_mb if x.est_kv_mb is not None else 1e30,
            )
            best_turbo = min(turbo_at_max if turbo_at_max else turbo_ok, key=lambda x: x.runtime_kv_mb)

            lines.append(
                f"- [{model}] At ctx={best_llama.context_tokens}, best llama.cpp KV memory: {best_llama.cache_type} with estimated {best_llama.est_kv_mb:.1f} MB KV."
            )
            lines.append(
                f"- [{model}] At ctx={best_turbo.context_tokens}, best TurboQuant runtime KV memory: policy={best_turbo.policy}, bits={best_turbo.bits}, runtime KV={best_turbo.runtime_kv_mb:.1f} MB."
            )

            if best_llama.est_kv_mb and best_turbo.runtime_kv_mb > 0:
                improvement = best_llama.est_kv_mb / best_turbo.runtime_kv_mb
                lines.append(f"- [{model}] KV memory improvement factor at matched context: {improvement:.2f}x.")

        unsupported_types = sorted(
            {
                r.cache_type
                for r in llama_rows
                if r.model_name == model
                and r.status in {"load-failed", "run-failed", "error"}
                and r.error
                and "Failed to create llama_context" in r.error
            }
        )
        if unsupported_types:
            lines.append(
                f"- [{model}] llama.cpp runtime did not accept quantized KV cache types: "
                + ", ".join(unsupported_types)
                + ". Compare these with caution unless llama.cpp is rebuilt with matching KV cache quantization support."
            )

    lines.append("- llama.cpp rows reflect end-to-end prompt/decode timing with requested KV cache type.")
    lines.append(
        "- TurboQuant rows benchmark compressed KV append and repeated attention-score queries; KV MB is scaled to all transformer layers using GGUF metadata."
    )
    lines.append(
        "- Estimated max context @ 8GB is a KV-only projection and does not include model weights, activations, or fragmentation overhead."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TurboQuant KV cache vs llama.cpp KV cache types for long-context memory and throughput"
    )
    parser.add_argument(
        "--gguf-model",
        action="append",
        default=[],
        help="Repeatable GGUF model path. If omitted, models are discovered from --gguf-root.",
    )
    parser.add_argument("--gguf-root", default="/mnt/c/models/gguf", help="Folder to auto-discover GGUF models")
    parser.add_argument("--all-models", action="store_true", help="Benchmark all discovered GGUF models")
    parser.add_argument("--model-include", default="qwen", help="Substring filter for discovered GGUF files")
    parser.add_argument("--model-exclude", default="mmproj", help="Exclude discovered GGUF files containing this substring")
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--contexts", nargs="*", type=int, default=[2048, 4096, 8192, 16384])
    parser.add_argument("--decode-tokens", type=int, default=12)
    parser.add_argument("--llama-cache-types", nargs="*", default=["f16", "q8_0", "q4_0"])
    parser.add_argument("--llama-n-gpu-layers", type=int, default=-1)
    parser.add_argument("--llama-threads", type=int, default=6)
    parser.add_argument("--llama-threads-batch", type=int, default=6)
    parser.add_argument("--llama-n-batch", type=int, default=256)
    parser.add_argument("--turboquant-bits", nargs="*", type=int, default=[2, 3, 4])
    parser.add_argument("--turboquant-policies", nargs="*", default=["packed", "prepared"])
    parser.add_argument("--score-backend", choices=["xla", "pallas"], default="xla")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--query-tile-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--vram-budget-mb", type=float, default=8192.0)
    parser.add_argument("--cooldown-s", type=float, default=1.0, help="Pause between heavy benchmark phases")
    parser.add_argument("--safe-mode", action="store_true", help="Use conservative settings to reduce system/GPU pressure")
    parser.add_argument("--safe-max-context", type=int, default=8192)
    parser.add_argument("--safe-decode-tokens", type=int, default=4)
    parser.add_argument("--safe-llama-n-gpu-layers", type=int, default=16)
    parser.add_argument("--safe-llama-threads", type=int, default=4)
    parser.add_argument("--safe-llama-threads-batch", type=int, default=4)
    parser.add_argument("--safe-llama-n-batch", type=int, default=128)
    parser.add_argument("--safe-turboquant-bits", nargs="*", type=int, default=[2, 3])
    parser.add_argument("--safe-turboquant-policies", nargs="*", default=["packed"])
    parser.add_argument("--json-output", default="", help="Optional path to dump raw JSON rows")
    args = parser.parse_args()

    contexts = sorted({int(c) for c in args.contexts if int(c) > 0})
    if not contexts:
        raise ValueError("At least one positive context length is required")

    run_profile = "default"
    if args.safe_mode:
        safe_contexts = [c for c in contexts if c <= int(args.safe_max_context)]
        if safe_contexts:
            contexts = safe_contexts
        args.decode_tokens = min(int(args.decode_tokens), int(args.safe_decode_tokens))
        if int(args.llama_n_gpu_layers) == -1:
            args.llama_n_gpu_layers = int(args.safe_llama_n_gpu_layers)
        args.llama_threads = min(int(args.llama_threads), int(args.safe_llama_threads))
        args.llama_threads_batch = min(int(args.llama_threads_batch), int(args.safe_llama_threads_batch))
        args.llama_n_batch = min(int(args.llama_n_batch), int(args.safe_llama_n_batch))
        args.turboquant_bits = list(args.safe_turboquant_bits)
        args.turboquant_policies = list(args.safe_turboquant_policies)
        args.cooldown_s = max(float(args.cooldown_s), 2.0)
        run_profile = (
            f"safe-mode contexts<={args.safe_max_context}, decode={args.decode_tokens}, "
            f"llama_layers={args.llama_n_gpu_layers}, llama_threads={args.llama_threads}, "
            f"llama_n_batch={args.llama_n_batch}, bits={args.turboquant_bits}, "
            f"policies={args.turboquant_policies}, cooldown={args.cooldown_s}s"
        )

    policies = [p.strip().lower() for p in args.turboquant_policies]
    for p in policies:
        if p not in {"packed", "prepared", "adaptive"}:
            raise ValueError(f"Unsupported TurboQuant policy: {p}")

    explicit_paths = [Path(p) for p in args.gguf_model]
    discovered_paths: list[Path] = []
    if args.all_models or not explicit_paths:
        discovered_paths = _discover_gguf_models(Path(args.gguf_root), args.model_include, args.model_exclude)

    model_paths = sorted({p for p in (explicit_paths + discovered_paths)})
    if not model_paths:
        raise FileNotFoundError("No GGUF models selected. Pass --gguf-model and/or adjust --gguf-root discovery filters.")

    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    model_runs: list[ModelRunInfo] = []
    llama_rows: list[LlamaKvRow] = []
    turbo_rows: list[TurboKvRow] = []

    total = len(model_paths)
    for idx, model_path in enumerate(model_paths, start=1):
        model_name = _infer_model_name_from_path(model_path)
        print(f"[model {idx}/{total}] Reading metadata for {model_name} ...", flush=True)
        dims = _infer_arch_dims(
            model_path,
            llama_threads=args.llama_threads,
            llama_threads_batch=args.llama_threads_batch,
            llama_n_batch=args.llama_n_batch,
        )
        model_runs.append(ModelRunInfo(model_name=model_name, model_path=model_path, dims=dims))

        print(f"[model {idx}/{total}] Running llama.cpp benchmark sweep for {model_name} ...", flush=True)
        llama_rows.extend(
            _benchmark_llama_cpp(
                model_name=model_name,
                model_path=model_path,
                contexts=contexts,
                decode_tokens=args.decode_tokens,
                n_gpu_layers=args.llama_n_gpu_layers,
                cache_types=args.llama_cache_types,
                dims=dims,
                vram_budget_mb=args.vram_budget_mb,
                llama_threads=args.llama_threads,
                llama_threads_batch=args.llama_threads_batch,
                llama_n_batch=args.llama_n_batch,
                run_cooldown_s=args.cooldown_s,
            )
        )

        print(f"[model {idx}/{total}] Running TurboQuant JAX benchmark sweep for {model_name} ...", flush=True)
        turbo_rows.extend(
            _benchmark_turboquant(
                model_name=model_name,
                contexts=contexts,
                decode_tokens=args.decode_tokens,
                dims=dims,
                bits_list=[int(b) for b in args.turboquant_bits],
                policies=policies,
                score_backend=args.score_backend,
                tile_size=args.tile_size,
                query_tile_size=args.query_tile_size,
                vram_budget_mb=args.vram_budget_mb,
                device=args.device,
                run_cooldown_s=args.cooldown_s,
            )
        )
        _cooldown(args.cooldown_s)

    print("[3/3] Writing markdown report section ...", flush=True)
    report_path = Path(args.report_path)
    section = _render_section(
        model_runs=model_runs,
        contexts=contexts,
        decode_tokens=args.decode_tokens,
        vram_budget_mb=args.vram_budget_mb,
        llama_rows=llama_rows,
        turbo_rows=turbo_rows,
        score_backend=args.score_backend,
        run_profile=run_profile,
    )
    _replace_section(report_path, SECTION_MARKER, section)

    if args.json_output:
        payload = {
            "model_runs": [
                {
                    "model_name": run.model_name,
                    "gguf_model": str(run.model_path),
                    "dims": run.dims,
                }
                for run in model_runs
            ],
            "contexts": contexts,
            "decode_tokens": args.decode_tokens,
            "vram_budget_mb": args.vram_budget_mb,
            "llama_rows": [r.__dict__ for r in llama_rows],
            "turbo_rows": [r.__dict__ for r in turbo_rows],
        }
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Updated report at {report_path}")


if __name__ == "__main__":
    main()

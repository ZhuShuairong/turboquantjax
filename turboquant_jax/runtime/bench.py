from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .generate import GenerateRequest, GenerationResult, build_repeated_prompt
from .telemetry import export_json, export_markdown_table


@dataclass
class BenchmarkConfig:
    backend: str
    model: str | None = None
    model_path: str | None = None
    runtime: str = "llamacpp-python"
    contexts: list[int] | None = None
    cache_modes: list[str] | None = None
    max_new_tokens: int = 32
    bits_k: int = 3
    bits_v: int = 2
    prompt: str | None = None
    prompt_file: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    device: str = "auto"
    dtype: str = "auto"
    export_json_path: str | None = None
    export_md_path: str | None = None
    extra: dict[str, Any] | None = None


def _benchmark_prompt_for_context(context_tokens: int) -> str:
    # For backend-agnostic benchmarking, keep the prompt deterministic.
    base = (
        "Long-context benchmark text. "
        "This sentence is repeated to measure prefill/decode behavior at scale. "
    )
    words_per_repeat = max(1, len(base.split()))
    repeats = max(1, context_tokens // words_per_repeat)
    return base * repeats


def _result_to_row(result: GenerationResult, context_tokens: int, bits_k: int, bits_v: int) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "model": result.model,
        "backend": result.backend,
        "runtime_mode": result.runtime_mode,
        "cache_mode": result.cache_mode,
        "context_length": context_tokens,
        "prompt_length": metrics.prompt_tokens,
        "generation_length": metrics.generated_tokens,
        "key_bits": bits_k,
        "value_bits": bits_v,
        "ttft_s": round(metrics.ttft_s, 6),
        "prefill_tps": round(metrics.prefill_tps, 3),
        "decode_tps": round(metrics.decode_tps, 3),
        "wall_time_s": round(metrics.wall_time_s, 6),
        "kv_cache_bytes": metrics.kv_cache_bytes,
        "peak_rss_mb": metrics.peak_rss_mb,
        "peak_gpu_mb": metrics.peak_gpu_mb,
        "compression_ratio": metrics.compression_ratio,
        "quality_score": metrics.quality_score,
    }


def run_benchmark(
    config: BenchmarkConfig,
    hf_backend,
    gguf_backend,
) -> list[dict[str, Any]]:
    contexts = config.contexts or [2048, 4096, 8192]
    cache_modes = config.cache_modes or ["baseline", "mse", "turboquant"]

    rows: list[dict[str, Any]] = []

    for context_tokens in contexts:
        prompt = config.prompt or _benchmark_prompt_for_context(context_tokens)

        for cache_mode in cache_modes:
            request = GenerateRequest(
                backend=config.backend,
                cache=cache_mode,
                model=config.model,
                model_path=config.model_path,
                runtime=config.runtime,
                prompt=prompt,
                prompt_file=config.prompt_file,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=config.seed,
                bits_k=config.bits_k,
                bits_v=config.bits_v,
                device=config.device,
                dtype=config.dtype,
                context_length=context_tokens,
                extra=dict(config.extra or {}),
            )

            if config.backend.lower().strip() == "hf":
                result = hf_backend.generate(request)
            elif config.backend.lower().strip() == "gguf":
                result = gguf_backend.generate(request)
            else:
                raise ValueError(f"Unsupported backend for benchmark: {config.backend}")

            rows.append(_result_to_row(result, context_tokens=context_tokens, bits_k=config.bits_k, bits_v=config.bits_v))

    if config.export_json_path:
        export_json(config.export_json_path, rows)
    if config.export_md_path:
        export_markdown_table(config.export_md_path, "TurboQuant Runtime Benchmark", rows)

    return rows

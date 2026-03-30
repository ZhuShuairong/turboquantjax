from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


SUPPORTED_BACKENDS = {"hf", "gguf"}
SUPPORTED_CACHE_MODES = {"baseline", "mse", "turboquant"}


@dataclass
class GenerateRequest:
    backend: str = "hf"
    cache: str = "baseline"
    model: str | None = None
    model_path: str | None = None
    runtime: str = "llamacpp-python"
    prompt: str | None = None
    prompt_file: str | None = None
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    bits_k: int = 3
    bits_v: int = 2
    device: str = "auto"
    dtype: str = "auto"
    context_length: int | None = None
    export_json: str | None = None
    export_md: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        backend = self.backend.lower().strip()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend}")
        cache = self.cache.lower().strip()
        if cache not in SUPPORTED_CACHE_MODES:
            raise ValueError(f"Unsupported cache mode: {self.cache}")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

        if backend == "hf" and not self.model:
            raise ValueError("HF backend requires --model")
        if backend == "gguf" and not self.model_path:
            raise ValueError("GGUF backend requires --model-path")

        if self.prompt is None and self.prompt_file is None:
            raise ValueError("Provide --prompt or --prompt-file")

    def prompt_text(self) -> str:
        if self.prompt is not None:
            return self.prompt
        if self.prompt_file is None:
            raise ValueError("No prompt source provided")
        return Path(self.prompt_file).read_text(encoding="utf-8")


@dataclass
class GenerationMetrics:
    ttft_s: float
    prefill_tps: float
    decode_tps: float
    wall_time_s: float
    prompt_tokens: int
    generated_tokens: int
    peak_rss_mb: float | None = None
    peak_gpu_mb: float | None = None
    kv_cache_bytes: int | None = None
    compression_ratio: float | None = None
    quality_score: float | None = None


@dataclass
class GenerationResult:
    backend: str
    model: str
    runtime_mode: str
    cache_mode: str
    prompt: str
    output_text: str
    metrics: GenerationMetrics
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "model": self.model,
            "runtime_mode": self.runtime_mode,
            "cache_mode": self.cache_mode,
            "prompt": self.prompt,
            "output_text": self.output_text,
            "metrics": {
                "ttft_s": self.metrics.ttft_s,
                "prefill_tps": self.metrics.prefill_tps,
                "decode_tps": self.metrics.decode_tps,
                "wall_time_s": self.metrics.wall_time_s,
                "prompt_tokens": self.metrics.prompt_tokens,
                "generated_tokens": self.metrics.generated_tokens,
                "peak_rss_mb": self.metrics.peak_rss_mb,
                "peak_gpu_mb": self.metrics.peak_gpu_mb,
                "kv_cache_bytes": self.metrics.kv_cache_bytes,
                "compression_ratio": self.metrics.compression_ratio,
                "quality_score": self.metrics.quality_score,
            },
            "metadata": self.metadata,
        }
        return payload


def build_repeated_prompt(
    target_tokens: int,
    tokenizer_encode: Callable[[str], list[int]],
    filler: str | None = None,
) -> str:
    text = ""
    base = filler or (
        "Long-context benchmark payload. "
        "This sentence is repeated to expand context while preserving deterministic content. "
    )

    while True:
        text += base
        if len(tokenizer_encode(text)) >= target_tokens:
            return text

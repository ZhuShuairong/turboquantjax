from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generate import GenerateRequest, GenerationMetrics, GenerationResult
from .llamacpp_bridge import LlamaCppBridge
from .telemetry import ResourceTracker


@dataclass
class GgufBackend:
    bridge: LlamaCppBridge

    def healthcheck(self, model_path: str | None = None) -> dict[str, Any]:
        path_ok = bool(model_path and Path(model_path).exists()) if model_path else None
        return {
            "llamacpp_python_installed": self.bridge.has_python_binding(),
            "llama_cli_available": self.bridge.has_binary("llama-cli"),
            "llama_server_available": self.bridge.has_binary("llama-server"),
            "model_path": model_path,
            "model_path_exists": path_ok,
            "ready": bool(self.bridge.has_python_binding() or self.bridge.has_binary("llama-cli")),
        }

    def generate(self, request: GenerateRequest) -> GenerationResult:
        request.validate()
        if request.backend.lower().strip() != "gguf":
            raise ValueError("GgufBackend only supports backend=gguf")

        model_path = request.model_path or ""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        prompt = request.prompt_text()
        runtime = request.runtime.lower().strip()

        with ResourceTracker() as tracker:
            if runtime == "llamacpp-python":
                payload = self.bridge.generate_with_python(
                    model_path=model_path,
                    prompt=prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    seed=request.seed,
                    context_length=request.context_length,
                    cache_type=str(request.extra.get("llama_cache_type", "f16")) if isinstance(request.extra, dict) else "f16",
                )
            elif runtime == "llamacpp-cli":
                payload = self.bridge.generate_with_cli(
                    model_path=model_path,
                    prompt=prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    context_length=request.context_length,
                    binary=str(request.extra.get("llama_cli_binary", "llama-cli")) if isinstance(request.extra, dict) else "llama-cli",
                )
            elif runtime == "llama-server":
                server_url = None
                if isinstance(request.extra, dict):
                    server_url = request.extra.get("server_url")
                if not server_url:
                    raise ValueError("runtime=llama-server requires extra['server_url']")
                payload = self.bridge.call_server(
                    server_url=str(server_url),
                    prompt=prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    model=request.model,
                )
            else:
                raise ValueError(f"Unsupported GGUF runtime: {request.runtime}")

        cache_mode = request.cache.lower().strip()
        cache_note = "baseline GGUF runtime"
        if cache_mode in {"mse", "turboquant"}:
            cache_note = (
                "staged integration: GGUF generation is executed by llama.cpp; "
                "TurboQuant cache mode currently reports staged capability and benchmark instrumentation path."
            )

        metrics = GenerationMetrics(
            ttft_s=float(payload.get("ttft_s", 0.0)),
            prefill_tps=float(payload.get("prefill_tps", 0.0)),
            decode_tps=float(payload.get("decode_tps", 0.0)),
            wall_time_s=float(payload.get("wall_time_s", 0.0)),
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            generated_tokens=int(payload.get("generated_tokens", 0)),
            peak_rss_mb=tracker.peak_rss_mb,
            peak_gpu_mb=tracker.peak_gpu_mb,
            kv_cache_bytes=None,
            compression_ratio=None,
            quality_score=None,
        )

        metadata: dict[str, Any] = {
            "cache_note": cache_note,
            "runtime_payload": payload,
        }

        return GenerationResult(
            backend="gguf",
            model=model_path,
            runtime_mode=runtime,
            cache_mode=cache_mode,
            prompt=prompt,
            output_text=str(payload.get("output_text", "")),
            metrics=metrics,
            metadata=metadata,
        )

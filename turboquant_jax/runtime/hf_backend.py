from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Any

from .cache import TurboQuantCacheAnalyzer
from .generate import GenerateRequest, GenerationMetrics, GenerationResult
from .telemetry import ResourceTracker


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_torch_dtype(dtype_name: str):
    import torch

    value = (dtype_name or "auto").lower().strip()
    if value == "fp16" or value == "float16":
        return torch.float16
    if value == "bf16" or value == "bfloat16":
        return torch.bfloat16
    if value == "fp32" or value == "float32":
        return torch.float32
    return "auto"


@dataclass
class HfBackend:
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = True
    local_files_only: bool = False

    def healthcheck(self, model: str | None = None) -> dict[str, Any]:
        checks = {
            "transformers_installed": _has_module("transformers"),
            "torch_installed": _has_module("torch"),
            "model": model,
        }

        if checks["torch_installed"]:
            import torch

            checks["torch_cuda_available"] = bool(torch.cuda.is_available())
        else:
            checks["torch_cuda_available"] = False

        checks["ready"] = bool(checks["transformers_installed"] and checks["torch_installed"])
        return checks

    def _load(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import torch

        torch_dtype = _resolve_torch_dtype(self.dtype)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if torch_dtype != "auto":
            load_kwargs["torch_dtype"] = torch_dtype

        if self.device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        else:
            # For GPU and multi-GPU, rely on accelerate device mapping.
            load_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model.eval()
        return model, tokenizer

    def _sample_next_token(self, logits, temperature: float, top_p: float, seed: int, step: int):
        import torch

        if temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumulative <= top_p
            keep_mask[..., 0] = True
            filtered = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)

            generator = torch.Generator(device=logits.device)
            generator.manual_seed(int(seed + step))
            sample = torch.multinomial(filtered, num_samples=1, generator=generator)
            token = sorted_idx.gather(-1, sample)
            return token

        generator = torch.Generator(device=logits.device)
        generator.manual_seed(int(seed + step))
        return torch.multinomial(probs, num_samples=1, generator=generator)

    def generate(self, request: GenerateRequest) -> GenerationResult:
        request.validate()
        if request.backend.lower().strip() != "hf":
            raise ValueError("HfBackend only supports backend=hf")

        model_name = request.model or ""
        prompt = request.prompt_text()

        model, tokenizer = self._load(model_name)

        import torch

        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=bool(request.context_length),
            max_length=request.context_length,
        )
        if self.device == "cpu":
            tokenized = tokenized.to("cpu")

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")

        prompt_tokens = int(input_ids.shape[1])

        with ResourceTracker() as tracker:
            # Prefill pass.
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
            prefill_s = time.perf_counter() - prefill_start

            past = out.past_key_values
            logits = out.logits[:, -1, :]

            generated_ids = []

            # First decode token defines TTFT after prefill.
            ttft_start = time.perf_counter()
            next_token = self._sample_next_token(logits, request.temperature, request.top_p, request.seed, step=0)
            generated_ids.append(next_token)

            with torch.inference_mode():
                out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            ttft_s = prefill_s + (time.perf_counter() - ttft_start)

            past = out.past_key_values
            logits = out.logits[:, -1, :]

            # Remaining decode steps.
            decode_start = time.perf_counter()
            for step in range(1, request.max_new_tokens):
                next_token = self._sample_next_token(logits, request.temperature, request.top_p, request.seed, step=step)
                generated_ids.append(next_token)
                with torch.inference_mode():
                    out = model(input_ids=next_token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
            decode_s = time.perf_counter() - decode_start

            generated = torch.cat(generated_ids, dim=1)
            output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

        cache_summary = None
        if request.cache.lower().strip() in {"mse", "turboquant"}:
            analyzer = TurboQuantCacheAnalyzer(bits_k=request.bits_k, bits_v=request.bits_v)
            cache_summary = analyzer.analyze_cache(
                past,
                mode=request.cache,
                max_layers=request.extra.get("max_eval_layers") if isinstance(request.extra, dict) else None,
            )

        generated_tokens = int(generated.shape[1])
        decode_tokens_for_rate = max(1, generated_tokens - 1)
        decode_elapsed = max(decode_s, 1e-9)
        metrics = GenerationMetrics(
            ttft_s=float(ttft_s),
            prefill_tps=float(prompt_tokens / max(prefill_s, 1e-9)),
            decode_tps=float(decode_tokens_for_rate / decode_elapsed),
            wall_time_s=float(prefill_s + decode_s),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            peak_rss_mb=tracker.peak_rss_mb,
            peak_gpu_mb=tracker.peak_gpu_mb,
            kv_cache_bytes=(cache_summary.compressed_bytes if cache_summary else None),
            compression_ratio=(cache_summary.compression_ratio if cache_summary else None),
            quality_score=(cache_summary.score_cosine_mean if cache_summary else None),
        )

        metadata: dict[str, Any] = {
            "device": self.device,
            "dtype": self.dtype,
            "cache_summary": cache_summary.to_dict() if cache_summary is not None else None,
        }

        return GenerationResult(
            backend="hf",
            model=model_name,
            runtime_mode="transformers",
            cache_mode=request.cache,
            prompt=prompt,
            output_text=output_text,
            metrics=metrics,
            metadata=metadata,
        )

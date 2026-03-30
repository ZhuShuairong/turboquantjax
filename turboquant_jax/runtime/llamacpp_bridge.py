from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CACHE_TYPE_ALIASES = {
    "f16": "GGML_TYPE_F16",
    "q8_0": "GGML_TYPE_Q8_0",
    "q4_0": "GGML_TYPE_Q4_0",
    "q4_1": "GGML_TYPE_Q4_1",
    "q5_0": "GGML_TYPE_Q5_0",
    "q5_1": "GGML_TYPE_Q5_1",
}


class LlamaCppBridge:
    def has_python_binding(self) -> bool:
        return importlib.util.find_spec("llama_cpp") is not None

    def has_binary(self, name: str) -> bool:
        return shutil.which(name) is not None

    def _resolve_cache_type(self, llama_cpp: Any, cache_type: str) -> int | None:
        alias = CACHE_TYPE_ALIASES.get(cache_type.lower().strip())
        if alias is None or not hasattr(llama_cpp, alias):
            return None
        return int(getattr(llama_cpp, alias))

    def generate_with_python(
        self,
        model_path: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        context_length: int | None = None,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        n_threads_batch: int | None = None,
        n_batch: int = 512,
        cache_type: str = "f16",
    ) -> dict[str, Any]:
        if not self.has_python_binding():
            raise RuntimeError("llama_cpp Python binding is not installed")

        import llama_cpp  # type: ignore

        Llama = llama_cpp.Llama
        ctx = int(context_length or 4096)
        type_enum = self._resolve_cache_type(llama_cpp, cache_type)

        init_kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": ctx,
            "n_gpu_layers": int(n_gpu_layers),
            "n_threads": int(n_threads or max(1, os.cpu_count() or 1)),
            "n_threads_batch": int(n_threads_batch or max(1, os.cpu_count() or 1)),
            "n_batch": int(n_batch),
            "seed": int(seed),
            "verbose": False,
        }
        if type_enum is not None:
            init_kwargs["type_k"] = type_enum
            init_kwargs["type_v"] = type_enum

        llm = Llama(**init_kwargs)

        try:
            prompt_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            prompt_token_count = len(prompt_tokens)

            # Prefill
            prefill_start = time.perf_counter()
            llm.reset()
            llm.eval(prompt_tokens)
            prefill_s = time.perf_counter() - prefill_start

            # Decode token-by-token for explicit decode TPS.
            generated_tokens: list[int] = []
            decode_start = time.perf_counter()
            for _ in range(max_new_tokens):
                token = int(
                    llm.sample(
                        temp=max(temperature, 0.0),
                        top_p=float(top_p),
                        top_k=40,
                        repeat_penalty=1.0,
                    )
                )
                generated_tokens.append(token)
                llm.eval([token])
            decode_s = time.perf_counter() - decode_start

            output_text = llm.detokenize(generated_tokens).decode("utf-8", errors="ignore")

            ttft_s = prefill_s + (decode_s / max_new_tokens if max_new_tokens > 0 else 0.0)
            return {
                "output_text": output_text,
                "prompt_tokens": prompt_token_count,
                "generated_tokens": len(generated_tokens),
                "ttft_s": ttft_s,
                "prefill_tps": prompt_token_count / max(prefill_s, 1e-9),
                "decode_tps": len(generated_tokens) / max(decode_s, 1e-9),
                "wall_time_s": prefill_s + decode_s,
                "runtime": "llamacpp-python",
                "cache_type": cache_type,
            }
        finally:
            if hasattr(llm, "close"):
                llm.close()

    def generate_with_cli(
        self,
        model_path: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        context_length: int | None = None,
        binary: str = "llama-cli",
    ) -> dict[str, Any]:
        if not self.has_binary(binary):
            raise RuntimeError(f"Required binary not found in PATH: {binary}")

        cmd = [
            binary,
            "-m",
            str(model_path),
            "-p",
            prompt,
            "-n",
            str(max_new_tokens),
            "--temp",
            str(temperature),
            "--top-p",
            str(top_p),
            "--no-display-prompt",
        ]
        if context_length is not None:
            cmd.extend(["--ctx-size", str(int(context_length))])

        start = time.perf_counter()
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start

        if completed.returncode != 0:
            raise RuntimeError(
                "llama-cli failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )

        output_text = completed.stdout
        prompt_tokens_est = max(1, len(prompt.split()))
        generated_tokens_est = max(1, len(output_text.split()))
        return {
            "output_text": output_text,
            "prompt_tokens": prompt_tokens_est,
            "generated_tokens": generated_tokens_est,
            "ttft_s": elapsed,
            "prefill_tps": 0.0,
            "decode_tps": generated_tokens_est / max(elapsed, 1e-9),
            "wall_time_s": elapsed,
            "runtime": "llamacpp-cli",
            "cache_type": "unknown",
        }

    def call_server(
        self,
        server_url: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        model: str | None = None,
    ) -> dict[str, Any]:
        url = server_url.rstrip("/") + "/v1/chat/completions"
        body = {
            "model": model or "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"llama-server HTTP error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to call llama-server: {exc}") from exc

        elapsed = time.perf_counter() - start
        choices = payload.get("choices", [])
        text = ""
        if choices:
            message = choices[0].get("message", {})
            text = message.get("content", "")

        usage = payload.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", max(1, len(prompt.split()))))
        completion_tokens = int(usage.get("completion_tokens", max(1, len(text.split()))))

        return {
            "output_text": text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": completion_tokens,
            "ttft_s": elapsed,
            "prefill_tps": 0.0,
            "decode_tps": completion_tokens / max(elapsed, 1e-9),
            "wall_time_s": elapsed,
            "runtime": "llama-server",
            "cache_type": "server-managed",
        }

    def launch_server(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        context_length: int = 4096,
        n_gpu_layers: int = -1,
        binary: str = "llama-server",
        extra_args: list[str] | None = None,
    ) -> subprocess.Popen:
        if not self.has_binary(binary):
            raise RuntimeError(f"Required binary not found in PATH: {binary}")

        cmd = [
            binary,
            "-m",
            str(Path(model_path)),
            "--host",
            host,
            "--port",
            str(int(port)),
            "--ctx-size",
            str(int(context_length)),
            "--n-gpu-layers",
            str(int(n_gpu_layers)),
        ]
        if extra_args:
            cmd.extend(extra_args)

        return subprocess.Popen(cmd)

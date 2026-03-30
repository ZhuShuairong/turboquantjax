from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _nvidia_smi_ok() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    completed = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    return completed.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TurboQuant runtime environment diagnostics")
    parser.add_argument("--hf-model", default=None, help="Optional HF model id/path to validate existence")
    parser.add_argument("--gguf-model", default=None, help="Optional GGUF model path to validate existence")
    parser.add_argument("--export-json", default=None, help="Optional path to write JSON diagnostics")
    args = parser.parse_args()

    payload = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "dependencies": {
            "jax": _has_module("jax"),
            "transformers": _has_module("transformers"),
            "torch": _has_module("torch"),
            "llama_cpp": _has_module("llama_cpp"),
            "psutil": _has_module("psutil"),
        },
        "executables": {
            "nvidia_smi": shutil.which("nvidia-smi"),
            "llama_cli": shutil.which("llama-cli"),
            "llama_server": shutil.which("llama-server"),
        },
        "gpu": {
            "nvidia_smi_ok": _nvidia_smi_ok(),
        },
        "models": {
            "hf_model": args.hf_model,
            "gguf_model": args.gguf_model,
            "gguf_model_exists": bool(args.gguf_model and Path(args.gguf_model).exists()),
        },
    }

    if payload["dependencies"]["jax"]:
        import jax

        payload["jax"] = {
            "devices": [str(d) for d in jax.devices()],
            "default_backend": jax.default_backend(),
        }

    if payload["dependencies"]["torch"]:
        import torch

        payload["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }

    text = json.dumps(payload, indent=2)
    print(text)

    if args.export_json:
        target = Path(args.export_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

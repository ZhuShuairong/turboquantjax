from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> int:
    print("Running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HF + GGUF benchmark matrix")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--gguf-model-path", default=None)
    parser.add_argument("--contexts", default="2048,4096,8192,16384")
    parser.add_argument("--hf-cache-modes", default="baseline,mse,turboquant")
    parser.add_argument("--gguf-cache-modes", default="baseline,turboquant")
    parser.add_argument("--gguf-runtime", default="llamacpp-python")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    rc = 0

    if args.hf_model:
        rc = _run(
            [
                args.python,
                str(script_dir / "benchmark_hf_qwen35.py"),
                "--model",
                args.hf_model,
                "--contexts",
                args.contexts,
                "--cache-modes",
                args.hf_cache_modes,
            ]
        )
        if rc != 0:
            return rc

    if args.gguf_model_path:
        rc = _run(
            [
                args.python,
                str(script_dir / "benchmark_gguf_qwen35.py"),
                "--model-path",
                args.gguf_model_path,
                "--runtime",
                args.gguf_runtime,
                "--contexts",
                args.contexts,
                "--cache-modes",
                args.gguf_cache_modes,
            ]
        )
        if rc != 0:
            return rc

    if not args.hf_model and not args.gguf_model_path:
        print("Nothing to run. Provide --hf-model and/or --gguf-model-path.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

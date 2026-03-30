from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_jax.runtime.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quality validation against baseline workflow")
    parser.add_argument("--backend", choices=["hf", "gguf"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--runtime", default="llamacpp-python")
    parser.add_argument("--cache", choices=["baseline", "mse", "turboquant"], default="turboquant")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--bits-k", type=int, default=3)
    parser.add_argument("--bits-v", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--export-json", default="artifacts/validation_report.json")
    parser.add_argument("--export-md", default="artifacts/validation_report.md")
    args = parser.parse_args()

    argv = [
        "validate",
        "--backend",
        args.backend,
        "--cache",
        args.cache,
        "--runtime",
        args.runtime,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--bits-k",
        str(args.bits_k),
        "--bits-v",
        str(args.bits_v),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--context-length",
        str(args.context_length),
        "--export-json",
        args.export_json,
        "--export-md",
        args.export_md,
    ]

    if args.model:
        argv.extend(["--model", args.model])
    if args.model_path:
        argv.extend(["--model-path", args.model_path])
    if args.server_url:
        argv.extend(["--server-url", args.server_url])

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

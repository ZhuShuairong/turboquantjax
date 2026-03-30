from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_jax.runtime.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HF Qwen3.5 inference through TurboQuant runtime")
    parser.add_argument("--model", required=True, help="HF model id or local model directory")
    parser.add_argument("--cache", choices=["baseline", "mse", "turboquant"], default="turboquant")
    parser.add_argument("--bits-k", type=int, default=3)
    parser.add_argument("--bits-v", type=int, default=2)
    parser.add_argument("--prompt", default="Explain TurboQuant KV compression in two paragraphs.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--export-json", default=None)
    parser.add_argument("--export-md", default=None)
    args = parser.parse_args()

    argv = [
        "generate",
        "--backend",
        "hf",
        "--model",
        args.model,
        "--cache",
        args.cache,
        "--bits-k",
        str(args.bits_k),
        "--bits-v",
        str(args.bits_v),
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    if args.export_json:
        argv.extend(["--export-json", args.export_json])
    if args.export_md:
        argv.extend(["--export-md", args.export_md])

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

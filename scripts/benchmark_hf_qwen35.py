from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_jax.runtime.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark HF Qwen3.5 contexts with TurboQuant runtime")
    parser.add_argument("--model", required=True)
    parser.add_argument("--contexts", default="2048,4096,8192,16384")
    parser.add_argument("--cache-modes", default="baseline,mse,turboquant")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=3)
    parser.add_argument("--bits-v", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--export-json", default="artifacts/hf_bench.json")
    parser.add_argument("--export-md", default="artifacts/hf_bench.md")
    parser.add_argument("--max-eval-layers", type=int, default=8)
    args = parser.parse_args()

    argv = [
        "bench",
        "--backend",
        "hf",
        "--model",
        args.model,
        "--contexts",
        args.contexts,
        "--cache-modes",
        args.cache_modes,
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
        "--max-eval-layers",
        str(args.max_eval_layers),
        "--export-json",
        args.export_json,
        "--export-md",
        args.export_md,
    ]
    if args.prompt:
        argv.extend(["--prompt", args.prompt])

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

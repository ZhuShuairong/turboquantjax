from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_jax.runtime.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GGUF Qwen3.5 contexts with TurboQuant runtime")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runtime", default="llamacpp-python", choices=["llamacpp-python", "llamacpp-cli", "llama-server"])
    parser.add_argument("--contexts", default="2048,4096,8192,16384")
    parser.add_argument("--cache-modes", default="baseline,turboquant")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--export-json", default="artifacts/gguf_bench.json")
    parser.add_argument("--export-md", default="artifacts/gguf_bench.md")
    args = parser.parse_args()

    argv = [
        "bench",
        "--backend",
        "gguf",
        "--model-path",
        args.model_path,
        "--runtime",
        args.runtime,
        "--contexts",
        args.contexts,
        "--cache-modes",
        args.cache_modes,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--export-json",
        args.export_json,
        "--export-md",
        args.export_md,
    ]
    if args.prompt:
        argv.extend(["--prompt", args.prompt])
    if args.server_url:
        argv.extend(["--server-url", args.server_url])

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

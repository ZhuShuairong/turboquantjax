from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_jax.runtime.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GGUF Qwen3.5 inference through TurboQuant runtime")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model")
    parser.add_argument("--runtime", default="llamacpp-python", choices=["llamacpp-python", "llamacpp-cli", "llama-server"])
    parser.add_argument("--cache", choices=["baseline", "mse", "turboquant"], default="baseline")
    parser.add_argument("--prompt", default="Summarize the benefits of local inference in 5 bullet points.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--export-json", default=None)
    parser.add_argument("--export-md", default=None)
    args = parser.parse_args()

    argv = [
        "generate",
        "--backend",
        "gguf",
        "--model-path",
        args.model_path,
        "--runtime",
        args.runtime,
        "--cache",
        args.cache,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--context-length",
        str(args.context_length),
    ]
    if args.server_url:
        argv.extend(["--server-url", args.server_url])
    if args.export_json:
        argv.extend(["--export-json", args.export_json])
    if args.export_md:
        argv.extend(["--export-md", args.export_md])

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

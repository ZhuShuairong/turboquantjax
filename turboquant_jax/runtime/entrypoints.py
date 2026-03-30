from __future__ import annotations

import sys

from .cli import main


def tqjax_generate_main() -> int:
    return main(["generate", *sys.argv[1:]])


def tqjax_bench_main() -> int:
    return main(["bench", *sys.argv[1:]])


def tqjax_validate_main() -> int:
    return main(["validate", *sys.argv[1:]])


def tqjax_serve_main() -> int:
    return main(["serve", *sys.argv[1:]])


def tqjax_env_main() -> int:
    return main(["env", *sys.argv[1:]])

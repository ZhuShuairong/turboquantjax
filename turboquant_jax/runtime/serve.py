from __future__ import annotations

import signal
from dataclasses import dataclass

from .llamacpp_bridge import LlamaCppBridge


@dataclass
class ServeConfig:
    backend: str
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8080
    context_length: int = 4096
    n_gpu_layers: int = -1
    binary: str = "llama-server"


def run_server(config: ServeConfig) -> int:
    backend = config.backend.lower().strip()
    if backend != "gguf":
        raise ValueError(
            "Only backend=gguf is currently supported for tqjax-serve. "
            "For HF serving, use the generation/benchmark CLI commands in this release."
        )

    bridge = LlamaCppBridge()
    process = bridge.launch_server(
        model_path=config.model_path,
        host=config.host,
        port=config.port,
        context_length=config.context_length,
        n_gpu_layers=config.n_gpu_layers,
        binary=config.binary,
    )

    def _stop(_signum, _frame):
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    return process.wait()

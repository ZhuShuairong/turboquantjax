"""PyTorch-style compatibility shim for turboquant-jax."""

from turboquant_jax import (
    LloydMaxCodebook,
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
    TurboQuantKVCache,
    TurboQuantMSE,
    TurboQuantProd,
    solve_lloyd_max,
)

__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantKVCache",
    "TurboQuantCompressorV2",
    "TurboQuantCompressorMSE",
    "LloydMaxCodebook",
    "solve_lloyd_max",
]

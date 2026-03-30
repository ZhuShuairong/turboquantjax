"""PyTorch-style compatibility shim for compressor imports."""

from turboquant_jax import TurboQuantCompressorMSE, TurboQuantCompressorV2

__all__ = ["TurboQuantCompressorV2", "TurboQuantCompressorMSE"]

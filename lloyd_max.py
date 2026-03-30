"""PyTorch-style compatibility shim for Lloyd-Max imports."""

from turboquant_jax import LloydMaxCodebook, solve_lloyd_max

__all__ = ["LloydMaxCodebook", "solve_lloyd_max"]

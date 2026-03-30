from __future__ import annotations

from turboquant_jax.fused_kernels import has_pallas, pallas_supported_on_active_backend


def is_pallas_available() -> bool:
    return bool(has_pallas() and pallas_supported_on_active_backend())

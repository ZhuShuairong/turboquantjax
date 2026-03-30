from __future__ import annotations

import jax.numpy as jnp

from turboquant_jax.compressors import TurboQuantCompressorV2JAX


def score_with_compressor(
    compressor: TurboQuantCompressorV2JAX,
    queries: jnp.ndarray,
    compressed_keys: dict,
) -> jnp.ndarray:
    return compressor.asymmetric_attention_scores(queries, compressed_keys)

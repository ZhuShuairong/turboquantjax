from __future__ import annotations

import jax
import jax.numpy as jnp

from turboquant_jax.compressors import JAXTurboQuantKVCache


def test_cache_append_score_and_decompress_values() -> None:
    key = jax.random.PRNGKey(123)
    k1, k2, k3 = jax.random.split(key, 3)

    keys = jax.random.normal(k1, (1, 2, 128, 64), dtype=jnp.float32)
    values = jax.random.normal(k2, (1, 2, 128, 64), dtype=jnp.float32)
    queries = jax.random.normal(k3, (1, 2, 1, 64), dtype=jnp.float32)

    cache = JAXTurboQuantKVCache(d_key=64, d_value=64, bits=3, score_cache_policy="prepared")
    cache.append(keys, values)

    scores = cache.attention_scores(queries)
    restored_values = cache.get_values()
    usage = cache.memory_usage_bits()

    assert scores.shape[-1] == 128
    assert restored_values.shape == values.shape
    assert usage["compression_ratio"] > 1.0
    assert usage["runtime_total_bits"] >= usage["total_bits"]

from __future__ import annotations

import jax.numpy as jnp

from turboquant_jax.lloyd_max import get_lloyd_max_codebook


def test_codebook_symmetry_and_bounds() -> None:
    cb = get_lloyd_max_codebook(128, 3)
    assert cb.centroids.shape[0] == 8
    assert cb.boundaries.shape[0] == 7

    # Symmetry and monotonic boundaries are core validity checks.
    assert float(jnp.abs(jnp.sum(cb.centroids))) < 0.05
    assert bool(jnp.all(cb.boundaries[1:] > cb.boundaries[:-1]))


def test_codebook_cache_consistency() -> None:
    cb_a = get_lloyd_max_codebook(64, 2)
    cb_b = get_lloyd_max_codebook(64, 2)
    assert cb_a is cb_b

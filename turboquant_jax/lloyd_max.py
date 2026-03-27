from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Tuple

import jax.numpy as jnp
from scipy import integrate


def beta_pdf(x: float, d: int) -> float:
    if abs(x) >= 1.0:
        return 0.0
    coeff = math.gamma(d / 2.0) / (math.sqrt(math.pi) * math.gamma((d - 1.0) / 2.0))
    return coeff * (1.0 - x * x) ** ((d - 3.0) / 2.0)


def gaussian_approx_pdf(x: float, d: int) -> float:
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2.0 * math.pi * sigma2)) * math.exp(-(x * x) / (2.0 * sigma2))


@lru_cache(maxsize=None)
def solve_lloyd_max(
    d: int,
    bits: int,
    use_exact: bool = False,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n_levels = 2 ** bits
    pdf = (lambda x: beta_pdf(x, d)) if use_exact else (lambda x: gaussian_approx_pdf(x, d))

    sigma = 1.0 / math.sqrt(d)
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3.0] + boundaries + [hi * 3.0]

        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            denominator, _ = integrate.quad(pdf, a, b)
            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return jnp.asarray(centroids, dtype=jnp.float32), jnp.asarray(boundaries, dtype=jnp.float32)


@dataclass(frozen=True)
class LloydMaxCodebook:
    d: int
    bits: int
    centroids: jnp.ndarray
    boundaries: jnp.ndarray


@lru_cache(maxsize=None)
def get_lloyd_max_codebook(d: int, bits: int, use_exact: bool = False) -> LloydMaxCodebook:
    centroids, boundaries = solve_lloyd_max(d, bits, use_exact=use_exact)
    return LloydMaxCodebook(d=d, bits=bits, centroids=centroids, boundaries=boundaries)

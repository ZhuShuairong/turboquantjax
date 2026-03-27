from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .quantization_utils import (
    generate_qjl_matrix,
    generate_rotation_matrix,
    get_lloyd_max_codebook,
    quantize_with_boundaries,
)


class TurboQuantMSEState(NamedTuple):
    d: int
    bits: int
    Pi: jnp.ndarray
    centroids: jnp.ndarray
    boundaries: jnp.ndarray


class TurboQuantProdState(NamedTuple):
    d: int
    bits: int
    mse_bits: int
    qjl_dim: int
    mse_state: TurboQuantMSEState
    S: jnp.ndarray


class CompressedProd(NamedTuple):
    mse_indices: jnp.ndarray
    qjl_signs: jnp.ndarray
    residual_norm: jnp.ndarray


def init_turboquant_mse(d: int, bits: int, seed: int = 42, use_exact: bool = False) -> TurboQuantMSEState:
    codebook = get_lloyd_max_codebook(d, bits, use_exact=use_exact)
    return TurboQuantMSEState(
        d=d,
        bits=bits,
        Pi=generate_rotation_matrix(d, seed=seed),
        centroids=codebook.centroids,
        boundaries=codebook.boundaries,
    )


def init_turboquant_prod(
    d: int,
    bits: int,
    qjl_dim: int | None = None,
    seed: int = 42,
    use_exact: bool = False,
) -> TurboQuantProdState:
    mse_bits = max(bits - 1, 1)
    qjl_dim = qjl_dim or d
    mse_state = init_turboquant_mse(d=d, bits=mse_bits, seed=seed, use_exact=use_exact)
    return TurboQuantProdState(
        d=d,
        bits=bits,
        mse_bits=mse_bits,
        qjl_dim=qjl_dim,
        mse_state=mse_state,
        S=generate_qjl_matrix(d=d, m=qjl_dim, seed=seed + 1),
    )


@jax.jit
def mse_quantize(state: TurboQuantMSEState, x: jnp.ndarray) -> jnp.ndarray:
    y = x @ state.Pi.T
    return quantize_with_boundaries(y, state.boundaries)


@jax.jit
def mse_dequantize(state: TurboQuantMSEState, indices: jnp.ndarray) -> jnp.ndarray:
    y_hat = state.centroids[indices]
    return y_hat @ state.Pi


@jax.jit
def mse_forward(state: TurboQuantMSEState, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    indices = mse_quantize(state, x)
    x_hat = mse_dequantize(state, indices)
    return x_hat, indices


@jax.jit
def mse_forward_batch(state: TurboQuantMSEState, x_batch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jax.vmap(lambda row: mse_forward(state, row))(x_batch)


@jax.jit
def prod_quantize(state: TurboQuantProdState, x: jnp.ndarray) -> CompressedProd:
    x_hat, mse_indices = mse_forward(state.mse_state, x)
    residual = x - x_hat
    residual_norm = jnp.linalg.norm(residual, axis=-1)
    projected = residual @ state.S.T
    qjl_signs = jnp.where(projected >= 0.0, 1.0, -1.0)
    return CompressedProd(mse_indices=mse_indices, qjl_signs=qjl_signs, residual_norm=residual_norm)


@jax.jit
def prod_inner_product(state: TurboQuantProdState, y: jnp.ndarray, compressed: CompressedProd) -> jnp.ndarray:
    x_mse = mse_dequantize(state.mse_state, compressed.mse_indices)
    term1 = jnp.sum(y * x_mse, axis=-1)
    y_projected = y @ state.S.T
    qjl_ip = jnp.sum(y_projected * compressed.qjl_signs, axis=-1)
    correction_scale = jnp.sqrt(jnp.pi / 2.0) / jnp.asarray(state.qjl_dim, dtype=jnp.float32)
    term2 = compressed.residual_norm * correction_scale * qjl_ip
    return term1 + term2


@jax.jit
def pairwise_inner_products(state: TurboQuantProdState, queries: jnp.ndarray, compressed: CompressedProd) -> jnp.ndarray:
    def one_query(query: jnp.ndarray) -> jnp.ndarray:
        query_broadcast = jnp.broadcast_to(query, compressed.mse_indices.shape)
        return prod_inner_product(state, query_broadcast, compressed)

    return jax.vmap(one_query)(queries)


def _calibration_mse(
    scale: jnp.ndarray,
    term1: jnp.ndarray,
    residual_norm: jnp.ndarray,
    qjl_ip: jnp.ndarray,
    target_inner: jnp.ndarray,
) -> jnp.ndarray:
    pred = term1 + scale * residual_norm * qjl_ip
    return jnp.mean((pred - target_inner) ** 2)


@jax.jit
def calibrate_qjl_scale_step(
    scale: jnp.ndarray,
    term1: jnp.ndarray,
    residual_norm: jnp.ndarray,
    qjl_ip: jnp.ndarray,
    target_inner: jnp.ndarray,
    lr: float = 1e-2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    grad_scale = jax.grad(_calibration_mse)(scale, term1, residual_norm, qjl_ip, target_inner)
    return scale - lr * grad_scale, grad_scale

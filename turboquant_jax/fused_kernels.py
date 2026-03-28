from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl

    _HAS_PALLAS = True
except Exception:
    pl = None
    _HAS_PALLAS = False


def has_pallas() -> bool:
    return _HAS_PALLAS


@lru_cache(maxsize=1)
def pallas_supported_on_active_backend() -> bool:
    if not _HAS_PALLAS:
        return False

    try:
        devices = jax.devices()
        if not devices:
            return False

        def probe_kernel(x_ref, o_ref):
            o_ref[...] = x_ref[...]

        x = jnp.ones((1, 1), dtype=jnp.float32)
        out_shape = jax.ShapeDtypeStruct((1, 1), jnp.float32)
        y = pl.pallas_call(probe_kernel, out_shape=out_shape, grid=(1,))(x)
        y.block_until_ready()
        return True
    except Exception:
        return False


def fused_term1_xla(
    queries_rot: jnp.ndarray,
    idx_tile: jnp.ndarray,
    centroids: jnp.ndarray,
    vec_tile: jnp.ndarray,
) -> jnp.ndarray:
    rotated_tile = centroids[idx_tile.astype(jnp.int32)]
    logits = jnp.matmul(queries_rot.astype(jnp.float32), jnp.swapaxes(rotated_tile, -2, -1))
    return logits * vec_tile[..., None, :]


def _pallas_matmul_2d(q: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    if not _HAS_PALLAS:
        return jnp.matmul(q, jnp.swapaxes(k, -2, -1))

    m, d = q.shape
    n = k.shape[0]

    bm = 32
    bn = 64
    bk = 32

    m_pad = ((m + bm - 1) // bm) * bm
    n_pad = ((n + bn - 1) // bn) * bn
    d_pad = ((d + bk - 1) // bk) * bk

    q_pad = jnp.pad(q.astype(jnp.float32), ((0, m_pad - m), (0, d_pad - d)))
    k_pad = jnp.pad(k.astype(jnp.float32), ((0, n_pad - n), (0, d_pad - d)))

    def kernel(q_ref, k_ref, o_ref):
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)

        offs_m = pid_m * bm + jnp.arange(bm)
        offs_n = pid_n * bn + jnp.arange(bn)

        acc = jnp.zeros((bm, bn), dtype=jnp.float32)

        for kk in range(0, d_pad, bk):
            offs_k = kk + jnp.arange(bk)

            q_block = q_ref[offs_m[:, None], offs_k[None, :]]
            k_block = k_ref[offs_n[:, None], offs_k[None, :]]
            acc = acc + jnp.matmul(q_block, jnp.swapaxes(k_block, -2, -1))

        o_ref[offs_m[:, None], offs_n[None, :]] = acc

    out_shape = jax.ShapeDtypeStruct((m_pad, n_pad), jnp.float32)
    grid = (pl.cdiv(m_pad, bm), pl.cdiv(n_pad, bn))

    out_pad = pl.pallas_call(kernel, out_shape=out_shape, grid=grid)(q_pad, k_pad)
    return out_pad[:m, :n]


def fused_term1_pallas(
    queries_rot: jnp.ndarray,
    idx_tile: jnp.ndarray,
    centroids: jnp.ndarray,
    vec_tile: jnp.ndarray,
) -> jnp.ndarray:
    if not _HAS_PALLAS:
        return fused_term1_xla(queries_rot, idx_tile, centroids, vec_tile)

    b, h, sq, d = queries_rot.shape
    tile = idx_tile.shape[2]

    rotated_tile = centroids[idx_tile.astype(jnp.int32)]

    q_flat = queries_rot.reshape((b * h, sq, d))
    k_flat = rotated_tile.reshape((b * h, tile, d))

    out_flat = jax.vmap(_pallas_matmul_2d, in_axes=(0, 0))(q_flat, k_flat)
    out = out_flat.reshape((b, h, sq, tile))
    return out * vec_tile[..., None, :]

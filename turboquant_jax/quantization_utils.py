from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .lloyd_max import LloydMaxCodebook, get_lloyd_max_codebook as _get_codebook


@lru_cache(maxsize=None)
def get_lloyd_max_codebook(d: int, bits: int, use_exact: bool = False) -> LloydMaxCodebook:
    return _get_codebook(d, bits, use_exact=use_exact)


def generate_rotation_matrix(d: int, seed: int = 42) -> jnp.ndarray:
    key = jax.random.PRNGKey(seed)
    gaussian = jax.random.normal(key, (d, d), dtype=jnp.float32)
    q, r = jnp.linalg.qr(gaussian)
    diag_sign = jnp.sign(jnp.diag(r))
    diag_sign = jnp.where(diag_sign == 0.0, 1.0, diag_sign)
    return q * diag_sign[jnp.newaxis, :]


def generate_qjl_matrix(d: int, m: int | None = None, seed: int = 42) -> jnp.ndarray:
    if m is None:
        m = d
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (m, d), dtype=jnp.float32)


def quantize_with_boundaries(values: jnp.ndarray, boundaries: jnp.ndarray) -> jnp.ndarray:
    return jnp.searchsorted(boundaries, values, side="left").astype(jnp.int32)


def pack_low_bit_values(values: jnp.ndarray, bits: int) -> Tuple[jnp.ndarray, Tuple[int, ...]]:
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    values_np = np.asarray(values, dtype=np.uint8)
    original_shape = tuple(values_np.shape)
    flat = values_np.reshape(-1).astype(np.uint64)
    if flat.size == 0:
        return jnp.asarray(np.empty((0,), dtype=np.uint8)), original_shape

    max_value = (1 << bits) - 1
    if flat.min() < 0 or flat.max() > max_value:
        raise ValueError(f"values must fit in {bits} bits")

    total_bits = flat.size * bits
    out = np.zeros((total_bits + 7) // 8, dtype=np.uint8)

    bit_index = 0
    for val in flat:
        for offset in range(bits):
            out[bit_index >> 3] |= ((int(val) >> offset) & 1) << (bit_index & 7)
            bit_index += 1

    return jnp.asarray(out), original_shape


def unpack_low_bit_values(packed: jnp.ndarray, bits: int, original_shape: Sequence[int]) -> jnp.ndarray:
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    packed_np = np.asarray(packed, dtype=np.uint8)
    total = int(np.prod(tuple(original_shape)))
    if total == 0:
        return jnp.asarray(np.empty(tuple(original_shape), dtype=np.uint8))

    vals = np.zeros(total, dtype=np.uint8)
    bit_index = 0
    for i in range(total):
        value = 0
        for offset in range(bits):
            byte = packed_np[bit_index >> 3]
            bit = (byte >> (bit_index & 7)) & 1
            value |= bit << offset
            bit_index += 1
        vals[i] = value

    return jnp.asarray(vals.reshape(tuple(original_shape)))


def pack_sign_bits(signs: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[int, ...]]:
    binary = (np.asarray(signs) >= 0).astype(np.uint8)
    return pack_low_bit_values(jnp.asarray(binary), 1)


def unpack_sign_bits(packed: jnp.ndarray, original_shape: Sequence[int]) -> jnp.ndarray:
    bits = unpack_low_bit_values(packed, 1, original_shape)
    return bits.astype(jnp.int8) * 2 - 1

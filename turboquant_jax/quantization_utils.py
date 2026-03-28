from __future__ import annotations

from functools import lru_cache
import math
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

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

    values_u8 = jnp.asarray(values, dtype=jnp.uint8)
    original_shape = tuple(values_u8.shape)
    flat = values_u8.reshape(-1).astype(jnp.uint16)
    if flat.size == 0:
        return jnp.zeros((0,), dtype=jnp.uint8), original_shape

    # Values are expected to come from quantization indices already bounded by bit-width.

    # Little-endian bit layout: per value we emit bit0, bit1, ... bit(bits-1).
    bit_offsets = jnp.arange(bits, dtype=jnp.uint16)
    bit_matrix = ((flat[:, None] >> bit_offsets[None, :]) & 1).astype(jnp.uint8)
    bits_flat = bit_matrix.reshape(-1)

    pad_bits = (-bits_flat.shape[0]) % 8
    bits_padded = jnp.pad(bits_flat, (0, pad_bits), mode="constant", constant_values=0)
    bytes_matrix = bits_padded.reshape(-1, 8).astype(jnp.uint16)

    byte_offsets = jnp.arange(8, dtype=jnp.uint16)
    packed = jnp.sum(bytes_matrix << byte_offsets[None, :], axis=1).astype(jnp.uint8)

    return packed, original_shape


def unpack_low_bit_values(packed: jnp.ndarray, bits: int, original_shape: Sequence[int]) -> jnp.ndarray:
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    packed_u8 = jnp.asarray(packed, dtype=jnp.uint8).reshape(-1)
    total = math.prod(tuple(original_shape))
    if total == 0:
        return jnp.zeros(tuple(original_shape), dtype=jnp.uint8)

    nbits = total * bits

    bit_offsets8 = jnp.arange(8, dtype=jnp.uint8)
    bit_matrix8 = ((packed_u8[:, None] >> bit_offsets8[None, :]) & 1).astype(jnp.uint8)
    bits_flat = bit_matrix8.reshape(-1)[:nbits]
    bit_matrix = bits_flat.reshape(total, bits).astype(jnp.uint16)

    bit_offsets = jnp.arange(bits, dtype=jnp.uint16)
    vals = jnp.sum(bit_matrix << bit_offsets[None, :], axis=1).astype(jnp.uint8)

    return vals.reshape(tuple(original_shape))


def unpack_low_bit_values_block(
    packed: jnp.ndarray,
    bits: int,
    start_value: jnp.ndarray,
    num_values: int,
) -> jnp.ndarray:
    """Decode a contiguous value block from a packed low-bit stream.

    The block starts at flattened value index ``start_value`` and decodes
    exactly ``num_values`` elements. This avoids unpacking the entire tensor
    when only one tile is needed.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")
    if num_values == 0:
        return jnp.zeros((0,), dtype=jnp.uint8)

    packed_u8 = jnp.asarray(packed, dtype=jnp.uint8).reshape(-1)
    start = jnp.asarray(start_value, dtype=jnp.int32)

    value_offsets = jnp.arange(num_values, dtype=jnp.int32)
    value_indices = start + value_offsets

    bit_offsets = jnp.arange(bits, dtype=jnp.int32)
    bit_positions = value_indices[:, None] * bits + bit_offsets[None, :]

    byte_indices = bit_positions >> 3
    bit_shifts = bit_positions & 7

    gathered = packed_u8[byte_indices]
    bits_matrix = ((gathered >> bit_shifts.astype(jnp.uint8)) & 1).astype(jnp.uint16)
    vals = jnp.sum(bits_matrix << bit_offsets[None, :].astype(jnp.uint16), axis=1)
    return vals.astype(jnp.uint8)


def pack_sign_bits(signs: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[int, ...]]:
    binary = (jnp.asarray(signs) >= 0).astype(jnp.uint8)
    return pack_low_bit_values(binary, 1)


def unpack_sign_bits(packed: jnp.ndarray, original_shape: Sequence[int]) -> jnp.ndarray:
    bits = unpack_low_bit_values(packed, 1, original_shape)
    return bits.astype(jnp.int8) * 2 - 1


def unpack_sign_bits_block(
    packed: jnp.ndarray,
    start_value: jnp.ndarray,
    num_values: int,
) -> jnp.ndarray:
    """Decode a contiguous block of packed sign bits and map to {-1, +1}."""
    bits = unpack_low_bit_values_block(packed, 1, start_value, num_values)
    return bits.astype(jnp.int8) * 2 - 1

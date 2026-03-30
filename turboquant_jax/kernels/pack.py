from __future__ import annotations

import jax.numpy as jnp

from turboquant_jax.quantization_utils import pack_low_bit_values, unpack_low_bit_values


def pack_values(values: jnp.ndarray, bits: int):
    return pack_low_bit_values(values, bits)


def unpack_values(packed: jnp.ndarray, bits: int, original_shape: tuple[int, ...]) -> jnp.ndarray:
    return unpack_low_bit_values(packed, bits, original_shape)

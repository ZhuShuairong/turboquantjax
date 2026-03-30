from __future__ import annotations

import jax.numpy as jnp

from turboquant_jax.kernels.pack import pack_values, unpack_values


def test_pack_unpack_roundtrip_multiple_bits() -> None:
    for bits in [1, 2, 3, 4, 7, 8]:
        values = jnp.arange(0, 257, dtype=jnp.uint16) % (2 ** bits)
        values = values.astype(jnp.uint8).reshape(257)
        packed, shape = pack_values(values, bits)
        restored = unpack_values(packed, bits, shape)
        assert restored.shape == values.shape
        assert bool(jnp.all(restored == values))


def test_pack_handles_empty_tensor() -> None:
    empty = jnp.zeros((0,), dtype=jnp.uint8)
    packed, shape = pack_values(empty, 3)
    restored = unpack_values(packed, 3, shape)
    assert tuple(shape) == (0,)
    assert restored.size == 0

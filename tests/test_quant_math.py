from __future__ import annotations

import jax
import jax.numpy as jnp

from turboquant_jax.turboquant import init_turboquant_mse, init_turboquant_prod, mse_dequantize, mse_quantize, prod_inner_product, prod_quantize


def test_mse_roundtrip_shape() -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (32, 128), dtype=jnp.float32)
    state = init_turboquant_mse(128, bits=3)

    indices = jax.vmap(mse_quantize, in_axes=(None, 0))(state, x)
    x_hat = jax.vmap(mse_dequantize, in_axes=(None, 0))(state, indices)

    assert indices.shape == x.shape
    assert x_hat.shape == x.shape


def test_prod_inner_product_bias_small() -> None:
    k1, k2 = jax.random.split(jax.random.PRNGKey(42), 2)
    x = jax.random.normal(k1, (512, 64), dtype=jnp.float32)
    y = jax.random.normal(k2, (512, 64), dtype=jnp.float32)

    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    y = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)

    state = init_turboquant_prod(64, bits=3)
    compressed = jax.vmap(prod_quantize, in_axes=(None, 0))(state, x)
    estimated = jax.vmap(prod_inner_product, in_axes=(None, 0, 0))(state, y, compressed)
    truth = jnp.sum(x * y, axis=-1)

    bias = float(jnp.mean(estimated - truth))
    assert abs(bias) < 0.08

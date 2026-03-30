from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import jax.numpy as jnp

from .compressors import TurboQuantCompressorMSEJAX, TurboQuantCompressorV2JAX
from .turboquant import (
    CompressedProd,
    init_turboquant_mse,
    init_turboquant_prod,
    mse_dequantize,
    mse_forward,
    mse_quantize,
    prod_inner_product,
    prod_quantize,
)


def _to_compressed_prod(compressed: CompressedProd | Mapping[str, Any]) -> CompressedProd:
    if isinstance(compressed, CompressedProd):
        return compressed
    if isinstance(compressed, Mapping):
        try:
            return CompressedProd(
                mse_indices=jnp.asarray(compressed["mse_indices"]),
                qjl_signs=jnp.asarray(compressed["qjl_signs"]),
                residual_norm=jnp.asarray(compressed["residual_norm"]),
            )
        except KeyError as exc:
            raise KeyError(f"Missing key in compressed dict: {exc}") from exc

    raise TypeError("compressed must be a CompressedProd or mapping with mse_indices/qjl_signs/residual_norm")


class TurboQuantMSE:
    """Class-style compatibility wrapper for the JAX MSE stage."""

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu", use_exact: bool = False):
        self.d = int(d)
        self.bits = int(bits)
        self.seed = int(seed)
        self.device = device
        self.state = init_turboquant_mse(self.d, self.bits, seed=self.seed, use_exact=use_exact)

        # Keep familiar attribute names from the PyTorch API.
        self.Pi = self.state.Pi
        self.centroids = self.state.centroids
        self.boundaries = self.state.boundaries

    def rotate(self, x: jnp.ndarray) -> jnp.ndarray:
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        return x_arr @ self.Pi.T

    def unrotate(self, y: jnp.ndarray) -> jnp.ndarray:
        y_arr = jnp.asarray(y, dtype=jnp.float32)
        return y_arr @ self.Pi

    def quantize(self, x: jnp.ndarray) -> jnp.ndarray:
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        return mse_quantize(self.state, x_arr)

    def dequantize(self, indices: jnp.ndarray) -> jnp.ndarray:
        idx = jnp.asarray(indices, dtype=jnp.int32)
        return mse_dequantize(self.state, idx)

    def forward(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        return mse_forward(self.state, x_arr)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.forward(x)


class TurboQuantProd:
    """Class-style compatibility wrapper for the JAX unbiased product estimator."""

    def __init__(
        self,
        d: int,
        bits: int,
        qjl_dim: int | None = None,
        seed: int = 42,
        device: str = "cpu",
        use_exact: bool = False,
    ):
        self.d = int(d)
        self.bits = int(bits)
        self.seed = int(seed)
        self.device = device
        self.state = init_turboquant_prod(self.d, self.bits, qjl_dim=qjl_dim, seed=self.seed, use_exact=use_exact)

        self.mse_bits = self.state.mse_bits
        self.qjl_dim = self.state.qjl_dim
        self.S = self.state.S

        # Expose an MSE quantizer instance for compatibility with the PyTorch API.
        self.mse = TurboQuantMSE(self.d, self.mse_bits, seed=self.seed, device=device, use_exact=use_exact)

    def quantize(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        compressed = prod_quantize(self.state, x_arr)
        return {
            "mse_indices": compressed.mse_indices,
            "qjl_signs": compressed.qjl_signs,
            "residual_norm": compressed.residual_norm,
        }

    def dequantize(self, compressed: CompressedProd | Mapping[str, Any]) -> jnp.ndarray:
        compressed_prod = _to_compressed_prod(compressed)
        return mse_dequantize(self.state.mse_state, compressed_prod.mse_indices)

    def inner_product(self, y: jnp.ndarray, compressed: CompressedProd | Mapping[str, Any]) -> jnp.ndarray:
        y_arr = jnp.asarray(y, dtype=jnp.float32)
        compressed_prod = _to_compressed_prod(compressed)
        return prod_inner_product(self.state, y_arr, compressed_prod)

    def forward(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return self.quantize(x)

    def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return self.forward(x)


class TurboQuantKVCache:
    """Class-style compatibility wrapper that mirrors the PyTorch KV cache API."""

    def __init__(self, d_key: int, d_value: int, bits: int = 3, seed: int = 42, device: str = "cpu"):
        self.d_key = int(d_key)
        self.d_value = int(d_value)
        self.bits = int(bits)
        self.seed = int(seed)
        self.device = device

        self.key_quantizer = TurboQuantProd(self.d_key, self.bits, seed=self.seed, device=device)
        self.value_quantizer = TurboQuantMSE(self.d_value, self.bits, seed=self.seed + 100, device=device)

        self.key_cache: list[dict[str, Any]] = []
        self.value_cache: list[dict[str, Any]] = []

    def append(self, keys: jnp.ndarray, values: jnp.ndarray) -> None:
        keys_arr = jnp.asarray(keys, dtype=jnp.float32)
        values_arr = jnp.asarray(values, dtype=jnp.float32)

        orig_key_shape = tuple(int(x) for x in keys_arr.shape)
        orig_value_shape = tuple(int(x) for x in values_arr.shape)

        flat_keys = keys_arr.reshape(-1, self.d_key)
        flat_values = values_arr.reshape(-1, self.d_value)

        compressed_keys = self.key_quantizer.quantize(flat_keys)
        value_indices = self.value_quantizer.quantize(flat_values)

        self.key_cache.append(
            {
                "mse_indices": compressed_keys["mse_indices"],
                "qjl_signs": compressed_keys["qjl_signs"],
                "residual_norm": compressed_keys["residual_norm"],
                "shape": orig_key_shape,
            }
        )
        self.value_cache.append(
            {
                "indices": value_indices,
                "shape": orig_value_shape,
            }
        )

    def attention_scores(self, queries: jnp.ndarray) -> jnp.ndarray:
        if not self.key_cache:
            return jnp.zeros((0,), dtype=jnp.float32)

        queries_arr = jnp.asarray(queries, dtype=jnp.float32)
        scores: list[jnp.ndarray] = []
        for cached in self.key_cache:
            score = self.key_quantizer.inner_product(queries_arr, cached)
            scores.append(jnp.ravel(score))

        return jnp.concatenate(scores, axis=-1)

    def get_values(self) -> jnp.ndarray:
        if not self.value_cache:
            return jnp.zeros((0,), dtype=jnp.float32)

        values: list[jnp.ndarray] = []
        for cached in self.value_cache:
            values.append(self.value_quantizer.dequantize(cached["indices"]))

        return jnp.concatenate(values, axis=0)

    def memory_usage_bits(self) -> dict[str, float]:
        n_keys = sum(int(c["mse_indices"].size) for c in self.key_cache) if self.key_cache else 0
        n_qjl = sum(int(c["qjl_signs"].size) for c in self.key_cache) if self.key_cache else 0
        n_norms = sum(int(c["residual_norm"].size) for c in self.key_cache) if self.key_cache else 0
        n_values = sum(int(c["indices"].size) for c in self.value_cache) if self.value_cache else 0

        key_bits = n_keys * self.key_quantizer.mse_bits + n_qjl + n_norms * 16
        value_bits = n_values * self.bits
        total_bits = key_bits + value_bits
        fp16_bits = (n_keys + n_values) * 16

        return {
            "key_bits": float(key_bits),
            "value_bits": float(value_bits),
            "total_bits": float(total_bits),
            "fp16_bits": float(fp16_bits),
            "compression_ratio": float(fp16_bits / total_bits) if total_bits > 0 else 0.0,
        }

    def __len__(self) -> int:
        if not self.key_cache:
            return 0
        return sum(int(c["mse_indices"].shape[0]) for c in self.key_cache)


@dataclass
class TurboQuantCompressorV2:
    """Compatibility wrapper with PyTorch-style class name and signature."""

    head_dim: int
    bits: int
    seed: int
    device: str = "cpu"
    qjl_dim: int | None = None
    tile_size: int = 256
    query_tile_size: int = 128
    score_backend: Literal["xla", "pallas"] = "xla"

    def __post_init__(self) -> None:
        self._impl = TurboQuantCompressorV2JAX(
            head_dim=self.head_dim,
            bits=self.bits,
            seed=self.seed,
            qjl_dim=self.qjl_dim,
            tile_size=self.tile_size,
            query_tile_size=self.query_tile_size,
            score_backend=self.score_backend,
        )

    def compress(self, states: jnp.ndarray) -> dict[str, Any]:
        states_arr = jnp.asarray(states, dtype=jnp.float32)
        return self._impl.compress(states_arr)

    def asymmetric_attention_scores(self, queries: jnp.ndarray, compressed: dict[str, Any], chunk_size: int | None = None) -> jnp.ndarray:
        # chunk_size is accepted for API compatibility; tiled execution is managed internally.
        _ = chunk_size
        queries_arr = jnp.asarray(queries, dtype=jnp.float32)
        return self._impl.asymmetric_attention_scores(queries_arr, compressed)


@dataclass
class TurboQuantCompressorMSE:
    """Compatibility wrapper with PyTorch-style class name and signature."""

    head_dim: int
    bits: int
    seed: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._impl = TurboQuantCompressorMSEJAX(
            head_dim=self.head_dim,
            bits=self.bits,
            seed=self.seed,
        )

    def compress(self, states: jnp.ndarray) -> dict[str, Any]:
        states_arr = jnp.asarray(states, dtype=jnp.float32)
        return self._impl.compress(states_arr)

    def decompress(self, compressed: dict[str, Any]) -> jnp.ndarray:
        return self._impl.decompress(compressed)

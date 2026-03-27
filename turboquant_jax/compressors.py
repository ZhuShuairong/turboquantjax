from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .quantization_utils import (
    generate_qjl_matrix,
    generate_rotation_matrix,
    get_lloyd_max_codebook,
    pack_low_bit_values,
    pack_sign_bits,
    quantize_with_boundaries,
    unpack_low_bit_values,
    unpack_sign_bits,
)


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if hasattr(value, "dtype") and hasattr(value, "size"):
        arr = np.asarray(value)
        return int(arr.nbytes)
    return 0


@dataclass
class TurboQuantCompressorV2JAX:
    head_dim: int
    bits: int
    seed: int
    qjl_dim: int | None = None

    def __post_init__(self) -> None:
        self.mse_bits = max(self.bits - 1, 1)
        self.qjl_dim = self.qjl_dim or self.head_dim

        self.Pi = generate_rotation_matrix(self.head_dim, seed=self.seed)
        self.PiT = self.Pi.T

        codebook = get_lloyd_max_codebook(self.head_dim, self.mse_bits)
        self.centroids = codebook.centroids
        self.boundaries = codebook.boundaries

        self.S = generate_qjl_matrix(self.head_dim, m=self.qjl_dim, seed=self.seed + 10000)
        self.ST = self.S.T

        self._compress_core = jax.jit(self._compress_core_fn)
        self._score_core = jax.jit(self._score_core_fn)

    def _compress_core_fn(self, states: jnp.ndarray):
        b, h, s, d = states.shape
        flat = states.reshape(-1, d).astype(jnp.float32)

        vec_norms = jnp.linalg.norm(flat, axis=-1, keepdims=True)
        flat_norm = flat / (vec_norms + 1e-8)

        rotated = flat_norm @ self.PiT
        indices = quantize_with_boundaries(rotated, self.boundaries).astype(jnp.uint8)

        reconstructed_rotated = self.centroids[indices.astype(jnp.int32)]
        k_mse = (reconstructed_rotated @ self.Pi) * vec_norms

        residual = flat - k_mse
        residual_norm = jnp.linalg.norm(residual, axis=-1)

        projected = residual @ self.ST
        signs = jnp.where(projected >= 0.0, 1, 0).astype(jnp.uint8)

        return indices, signs, vec_norms.squeeze(-1).astype(jnp.float16), residual_norm.astype(jnp.float16), (b, h, s, d)

    def compress(self, states: jnp.ndarray) -> dict[str, Any]:
        indices, signs, vec_norms, residual_norm, shape = self._compress_core(states)
        packed_indices, indices_shape = pack_low_bit_values(indices, self.mse_bits)
        packed_signs, sign_shape = pack_sign_bits(signs)
        return {
            "indices": packed_indices,
            "indices_shape": indices_shape,
            "vec_norms": vec_norms,
            "qjl_signs": packed_signs,
            "qjl_sign_shape": sign_shape,
            "residual_norm": residual_norm,
            "shape": tuple(shape),
        }

    def _score_core_fn(
        self,
        queries: jnp.ndarray,
        indices: jnp.ndarray,
        signs: jnp.ndarray,
        vec_norms: jnp.ndarray,
        residual_norm: jnp.ndarray,
    ) -> jnp.ndarray:
        b, h, sq, d = queries.shape
        _, _, sk, _ = indices.shape

        q_projected = queries @ self.ST
        correction_scale = jnp.sqrt(jnp.pi / 2.0) / jnp.asarray(self.qjl_dim, dtype=jnp.float32)

        chunk_rotated = self.centroids[indices.astype(jnp.int32)]
        chunk_keys = (chunk_rotated @ self.Pi) * vec_norms[..., None]

        term1 = jnp.matmul(queries.astype(jnp.float32), jnp.swapaxes(chunk_keys, -2, -1))
        qjl_ip = jnp.matmul(q_projected, jnp.swapaxes(signs.astype(jnp.float32) * 2.0 - 1.0, -2, -1))
        term2 = correction_scale * qjl_ip * residual_norm[..., None, :]
        return term1 + term2

    def asymmetric_attention_scores(self, queries: jnp.ndarray, compressed: dict[str, Any]) -> jnp.ndarray:
        b, h, sk, d = compressed["shape"]
        indices = unpack_low_bit_values(compressed["indices"], self.mse_bits, compressed["indices_shape"]).astype(jnp.int32)
        indices = indices.reshape((b, h, sk, d))

        signs = unpack_sign_bits(compressed["qjl_signs"], compressed["qjl_sign_shape"]).astype(jnp.float32)
        signs = signs.reshape((b, h, sk, self.qjl_dim))

        vec_norms = jnp.asarray(compressed["vec_norms"], dtype=jnp.float32).reshape((b, h, sk))
        residual_norm = jnp.asarray(compressed["residual_norm"], dtype=jnp.float32).reshape((b, h, sk))

        return self._score_core(queries.astype(jnp.float32), indices, signs, vec_norms, residual_norm)


@dataclass
class TurboQuantCompressorMSEJAX:
    head_dim: int
    bits: int
    seed: int

    def __post_init__(self) -> None:
        self.Pi = generate_rotation_matrix(self.head_dim, seed=self.seed)
        self.PiT = self.Pi.T
        codebook = get_lloyd_max_codebook(self.head_dim, self.bits)
        self.centroids = codebook.centroids
        self.boundaries = codebook.boundaries
        self._compress_core = jax.jit(self._compress_core_fn)
        self._decompress_core = self._decompress_core_fn

    def _compress_core_fn(self, states: jnp.ndarray):
        b, h, s, d = states.shape
        flat = states.reshape(-1, d).astype(jnp.float32)
        vec_norms = jnp.linalg.norm(flat, axis=-1, keepdims=True)
        flat_norm = flat / (vec_norms + 1e-8)
        rotated = flat_norm @ self.PiT
        indices = quantize_with_boundaries(rotated, self.boundaries).astype(jnp.uint8)
        return indices, vec_norms.squeeze(-1).astype(jnp.float16), (b, h, s, d)

    def compress(self, states: jnp.ndarray) -> dict[str, Any]:
        indices, vec_norms, shape = self._compress_core(states)
        packed_indices, indices_shape = pack_low_bit_values(indices, self.bits)
        return {
            "indices": packed_indices,
            "indices_shape": indices_shape,
            "vec_norms": vec_norms,
            "shape": tuple(shape),
        }

    def _decompress_core_fn(self, indices: jnp.ndarray, vec_norms: jnp.ndarray, shape: tuple[int, int, int, int]) -> jnp.ndarray:
        b, h, s, d = shape
        reconstructed = self.centroids[indices.astype(jnp.int32)] @ self.Pi
        return (reconstructed * vec_norms[..., None]).reshape((b, h, s, d))

    def decompress(self, compressed: dict[str, Any]) -> jnp.ndarray:
        shape = tuple(compressed["shape"])
        indices = unpack_low_bit_values(compressed["indices"], self.bits, compressed["indices_shape"]).astype(jnp.int32)
        vec_norms = jnp.asarray(compressed["vec_norms"], dtype=jnp.float32)
        return self._decompress_core(indices, vec_norms, shape)


@dataclass
class JAXTurboQuantKVCache:
    d_key: int
    d_value: int
    bits: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        self.key_compressor = TurboQuantCompressorV2JAX(self.d_key, self.bits, self.seed)
        self.value_compressor = TurboQuantCompressorMSEJAX(self.d_value, self.bits, self.seed + 100)
        self.key_cache: list[dict[str, Any]] = []
        self.value_cache: list[dict[str, Any]] = []

    def append(self, keys: jnp.ndarray, values: jnp.ndarray) -> None:
        self.key_cache.append(self.key_compressor.compress(keys))
        self.value_cache.append(self.value_compressor.compress(values))

    def attention_scores(self, queries: jnp.ndarray) -> jnp.ndarray:
        scores = [self.key_compressor.asymmetric_attention_scores(queries, c) for c in self.key_cache]
        if not scores:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.concatenate(scores, axis=-1)

    def get_values(self) -> jnp.ndarray:
        values = [self.value_compressor.decompress(c) for c in self.value_cache]
        if not values:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.concatenate(values, axis=2)

    def memory_usage_bits(self) -> dict[str, float]:
        key_bytes = sum(_tensor_bytes(c) for c in self.key_cache)
        value_bytes = sum(_tensor_bytes(c) for c in self.value_cache)
        total_bytes = key_bytes + value_bytes

        fp16_bytes = 0
        for c in self.key_cache:
            b, h, s, d = c["shape"]
            fp16_bytes += b * h * s * d * 2
        for c in self.value_cache:
            b, h, s, d = c["shape"]
            fp16_bytes += b * h * s * d * 2

        return {
            "key_bits": float(key_bytes * 8),
            "value_bits": float(value_bytes * 8),
            "total_bits": float(total_bytes * 8),
            "fp16_bits": float(fp16_bytes * 8),
            "compression_ratio": float(fp16_bytes / total_bytes) if total_bytes > 0 else 0.0,
        }

    def __len__(self) -> int:
        total = 0
        for c in self.key_cache:
            b, h, s, _ = c["shape"]
            total += int(b * h * s)
        return total

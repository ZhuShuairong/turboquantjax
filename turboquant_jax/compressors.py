from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .fused_kernels import fused_term1_pallas, fused_term1_xla, has_pallas, pallas_supported_on_active_backend

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
    tile_size: int = 256
    score_backend: Literal["xla", "pallas"] = "xla"

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

        self._pallas_available = has_pallas() and pallas_supported_on_active_backend()
        if self.score_backend == "pallas" and not self._pallas_available:
            self.score_backend = "xla"

        self._compress_core = jax.jit(self._compress_core_fn)
        self._score_core_tiled = jax.jit(self._score_core_tiled_fn)

    def prepare_for_scoring(self, compressed: dict[str, Any]) -> dict[str, Any]:
        b, h, sk, d = compressed["shape"]
        indices = unpack_low_bit_values(compressed["indices"], self.mse_bits, compressed["indices_shape"]).astype(jnp.uint8)
        indices = indices.reshape((b, h, sk, d))

        signs = unpack_sign_bits(compressed["qjl_signs"], compressed["qjl_sign_shape"]).astype(jnp.int8)
        signs = signs.reshape((b, h, sk, self.qjl_dim))

        vec_norms = jnp.asarray(compressed["vec_norms"], dtype=jnp.float16).reshape((b, h, sk))
        residual_norm = jnp.asarray(compressed["residual_norm"], dtype=jnp.float16).reshape((b, h, sk))

        pad_tokens = (-sk) % self.tile_size if self.tile_size > 0 else 0
        if pad_tokens > 0:
            indices = jnp.pad(indices, ((0, 0), (0, 0), (0, pad_tokens), (0, 0)), mode="constant", constant_values=0)
            signs = jnp.pad(signs, ((0, 0), (0, 0), (0, pad_tokens), (0, 0)), mode="constant", constant_values=0)
            vec_norms = jnp.pad(vec_norms, ((0, 0), (0, 0), (0, pad_tokens)), mode="constant", constant_values=0.0)
            residual_norm = jnp.pad(residual_norm, ((0, 0), (0, 0), (0, pad_tokens)), mode="constant", constant_values=0.0)

        return {
            "indices": indices,
            "signs": signs,
            "vec_norms": vec_norms,
            "residual_norm": residual_norm,
            "shape": tuple(indices.shape),
            "valid_sk": sk,
            "pad_tokens": pad_tokens,
        }

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
        shape_ints = tuple(int(x) for x in shape)
        return {
            "indices": packed_indices,
            "indices_shape": indices_shape,
            "vec_norms": vec_norms,
            "qjl_signs": packed_signs,
            "qjl_sign_shape": sign_shape,
            "residual_norm": residual_norm,
            "shape": shape_ints,
        }

    def _score_core_tiled_fn(
        self,
        queries: jnp.ndarray,
        indices: jnp.ndarray,
        signs: jnp.ndarray,
        vec_norms: jnp.ndarray,
        residual_norm: jnp.ndarray,
    ) -> jnp.ndarray:
        b, h, sq, d = queries.shape
        _, _, sk, _ = indices.shape

        queries = queries.astype(jnp.float32)
        q_projected = queries @ self.ST
        q_rot = queries @ self.PiT
        correction_scale = jnp.sqrt(jnp.pi / 2.0) / jnp.asarray(self.qjl_dim, dtype=jnp.float32)
        tile = max(1, self.tile_size)
        n_tiles = sk // tile
        out_scores = jnp.zeros((b, h, sq, sk), dtype=jnp.float32)

        def body_fn(i: int, carry: jnp.ndarray) -> jnp.ndarray:
            start = i * tile
            idx_tile = jax.lax.dynamic_slice(indices, (0, 0, start, 0), (b, h, tile, d))
            sign_tile = jax.lax.dynamic_slice(signs, (0, 0, start, 0), (b, h, tile, self.qjl_dim))
            vec_tile = jax.lax.dynamic_slice(vec_norms, (0, 0, start), (b, h, tile))
            norm_tile = jax.lax.dynamic_slice(residual_norm, (0, 0, start), (b, h, tile))

            if self.score_backend == "pallas" and self._pallas_available:
                term1 = fused_term1_pallas(q_rot, idx_tile, self.centroids, vec_tile)
            else:
                term1 = fused_term1_xla(q_rot, idx_tile, self.centroids, vec_tile)
            qjl_ip = jnp.matmul(q_projected, jnp.swapaxes(sign_tile.astype(jnp.float32), -2, -1))
            term2 = correction_scale * qjl_ip * norm_tile.astype(jnp.float32)[..., None, :]
            tile_scores = term1 + term2
            return jax.lax.dynamic_update_slice(carry, tile_scores, (0, 0, 0, start))

        return jax.lax.fori_loop(0, n_tiles, body_fn, out_scores)

    def asymmetric_attention_scores_prepared(self, queries: jnp.ndarray, prepared: dict[str, Any]) -> jnp.ndarray:
        full_scores = self._score_core_tiled(
            queries.astype(jnp.float32),
            prepared["indices"],
            prepared["signs"],
            prepared["vec_norms"],
            prepared["residual_norm"],
        )
        return full_scores[..., : prepared["valid_sk"]]

    def asymmetric_attention_scores(self, queries: jnp.ndarray, compressed: dict[str, Any]) -> jnp.ndarray:
        prepared = self.prepare_for_scoring(compressed)
        return self.asymmetric_attention_scores_prepared(queries, prepared)


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
        self._decompress_core = jax.jit(self._decompress_core_fn, static_argnums=(2,))

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
        shape_ints = tuple(int(x) for x in shape)
        return {
            "indices": packed_indices,
            "indices_shape": indices_shape,
            "vec_norms": vec_norms,
            "shape": shape_ints,
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
    score_cache_policy: Literal["prepared", "packed", "adaptive"] = "prepared"
    score_backend: Literal["xla", "pallas"] = "xla"
    tile_size: int = 256
    adaptive_promote_after: int = 2
    adaptive_hit_rate_threshold: float = 0.35
    adaptive_memory_budget_mb: float | None = None

    def __post_init__(self) -> None:
        if self.score_cache_policy not in {"prepared", "packed", "adaptive"}:
            raise ValueError("score_cache_policy must be 'prepared', 'packed', or 'adaptive'")
        self.key_compressor = TurboQuantCompressorV2JAX(
            self.d_key,
            self.bits,
            self.seed,
            tile_size=self.tile_size,
            score_backend=self.score_backend,
        )
        self.value_compressor = TurboQuantCompressorMSEJAX(self.d_value, self.bits, self.seed + 100)
        self.key_cache: list[dict[str, Any]] = []
        self.prepared_key_cache: list[dict[str, Any] | None] = []
        self.value_cache: list[dict[str, Any]] = []
        self._reuse_counts: list[int] = []

        self._attention_calls = 0
        self._cache_hit_calls = 0
        self._prepared_score_calls = 0
        self._packed_score_calls = 0
        self._promotion_events = 0
        self._eviction_events = 0
        self._step = 0
        self._last_use_step: list[int] = []

    def _prepared_budget_bytes(self) -> float | None:
        if self.adaptive_memory_budget_mb is None:
            return None
        return float(self.adaptive_memory_budget_mb) * 1024.0 * 1024.0

    def _prepared_bytes_used(self) -> int:
        return sum(_tensor_bytes(c) for c in self.prepared_key_cache if c is not None)

    def _cache_hit_rate(self) -> float:
        if self._attention_calls <= 0:
            return 0.0
        return float(self._cache_hit_calls) / float(self._attention_calls)

    def _evict_prepared_to_budget(self, keep_idx: int | None = None) -> None:
        budget = self._prepared_budget_bytes()
        if budget is None:
            return

        while float(self._prepared_bytes_used()) > budget:
            candidates: list[tuple[int, int, int]] = []
            for i, prepared in enumerate(self.prepared_key_cache):
                if prepared is None:
                    continue
                if keep_idx is not None and i == keep_idx:
                    continue
                candidates.append((self._reuse_counts[i], self._last_use_step[i], i))

            if not candidates:
                break

            _, _, evict_idx = min(candidates, key=lambda x: (x[0], x[1]))
            self.prepared_key_cache[evict_idx] = None
            self._eviction_events += 1

    def _should_promote_prefix(self, idx: int) -> bool:
        if self.prepared_key_cache[idx] is not None:
            return False
        if self._reuse_counts[idx] < self.adaptive_promote_after:
            return False
        if self._cache_hit_rate() < self.adaptive_hit_rate_threshold:
            return False
        return True

    def _maybe_prepare_prefix(self, idx: int) -> dict[str, Any] | None:
        if not self._should_promote_prefix(idx):
            return self.prepared_key_cache[idx]

        prepared = self.key_compressor.prepare_for_scoring(self.key_cache[idx])
        self.prepared_key_cache[idx] = prepared
        self._promotion_events += 1

        self._evict_prepared_to_budget(keep_idx=idx)
        if self.prepared_key_cache[idx] is None:
            return None

        return prepared

    def append(self, keys: jnp.ndarray, values: jnp.ndarray) -> None:
        compressed_k = self.key_compressor.compress(keys)
        self.key_cache.append(compressed_k)
        if self.score_cache_policy == "prepared":
            self.prepared_key_cache.append(self.key_compressor.prepare_for_scoring(compressed_k))
        else:
            self.prepared_key_cache.append(None)
        self._reuse_counts.append(0)
        self._last_use_step.append(0)
        self.value_cache.append(self.value_compressor.compress(values))

    def attention_scores(self, queries: jnp.ndarray) -> jnp.ndarray:
        self._attention_calls += 1
        self._step += 1
        if self.key_cache:
            self._cache_hit_calls += 1

        scores: list[jnp.ndarray] = []
        if self.score_cache_policy == "prepared":
            for i, prepared in enumerate(self.prepared_key_cache):
                if prepared is None:
                    prepared = self.key_compressor.prepare_for_scoring(self.key_cache[i])
                    self.prepared_key_cache[i] = prepared
                scores.append(self.key_compressor.asymmetric_attention_scores_prepared(queries, prepared))
                self._prepared_score_calls += 1
                self._reuse_counts[i] += 1
                self._last_use_step[i] = self._step
        elif self.score_cache_policy == "packed":
            for i, compressed in enumerate(self.key_cache):
                scores.append(self.key_compressor.asymmetric_attention_scores(queries, compressed))
                self._packed_score_calls += 1
                self._reuse_counts[i] += 1
                self._last_use_step[i] = self._step
        else:
            for i, compressed in enumerate(self.key_cache):
                self._reuse_counts[i] += 1
                self._last_use_step[i] = self._step
                prepared = self._maybe_prepare_prefix(i)
                if prepared is not None:
                    scores.append(self.key_compressor.asymmetric_attention_scores_prepared(queries, prepared))
                    self._prepared_score_calls += 1
                else:
                    scores.append(self.key_compressor.asymmetric_attention_scores(queries, compressed))
                    self._packed_score_calls += 1

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
        prepared_key_bytes = sum(_tensor_bytes(c) for c in self.prepared_key_cache if c is not None)
        value_bytes = sum(_tensor_bytes(c) for c in self.value_cache)
        total_bytes = key_bytes + value_bytes
        runtime_total_bytes = total_bytes + prepared_key_bytes

        fp16_bytes = 0
        for c in self.key_cache:
            b, h, s, d = c["shape"]
            fp16_bytes += b * h * s * d * 2
        for c in self.value_cache:
            b, h, s, d = c["shape"]
            fp16_bytes += b * h * s * d * 2

        return {
            "key_bits": float(key_bytes * 8),
            "prepared_key_bits": float(prepared_key_bytes * 8),
            "value_bits": float(value_bytes * 8),
            "total_bits": float(total_bytes * 8),
            "runtime_total_bits": float(runtime_total_bytes * 8),
            "fp16_bits": float(fp16_bytes * 8),
            "compression_ratio": float(fp16_bytes / total_bytes) if total_bytes > 0 else 0.0,
            "runtime_compression_ratio": float(fp16_bytes / runtime_total_bytes) if runtime_total_bytes > 0 else 0.0,
            "score_cache_policy": self.score_cache_policy,
            "score_backend": self.key_compressor.score_backend,
            "prepared_cache_share": float(prepared_key_bytes / runtime_total_bytes) if runtime_total_bytes > 0 else 0.0,
        }

    def policy_stats(self) -> dict[str, float]:
        total_segments = len(self.key_cache)
        prepared_segments = sum(1 for c in self.prepared_key_cache if c is not None)
        avg_reuse = (sum(self._reuse_counts) / total_segments) if total_segments > 0 else 0.0

        return {
            "attention_calls": float(self._attention_calls),
            "cache_hit_rate": self._cache_hit_rate(),
            "prepared_score_calls": float(self._prepared_score_calls),
            "packed_score_calls": float(self._packed_score_calls),
            "promotion_events": float(self._promotion_events),
            "eviction_events": float(self._eviction_events),
            "prepared_segments": float(prepared_segments),
            "total_segments": float(total_segments),
            "avg_reuse_per_segment": float(avg_reuse),
            "adaptive_promote_after": float(self.adaptive_promote_after),
            "adaptive_hit_rate_threshold": float(self.adaptive_hit_rate_threshold),
        }

    def __len__(self) -> int:
        total = 0
        for c in self.key_cache:
            b, h, s, _ = c["shape"]
            total += int(b * h * s)
        return total

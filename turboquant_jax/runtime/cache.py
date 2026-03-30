from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import jax.numpy as jnp
import numpy as np

from turboquant_jax.compressors import TurboQuantCompressorMSEJAX, TurboQuantCompressorV2JAX


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)
        except Exception:
            return 0
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        try:
            return int(value.numel() * value.element_size())
        except Exception:
            return 0
    return 0


def extract_cache_layers(cache_obj: Any) -> list[tuple[Any, Any]]:
    """Return list[(keys, values)] from multiple HF cache formats."""
    if cache_obj is None:
        return []

    if hasattr(cache_obj, "layers"):
        out: list[tuple[Any, Any]] = []
        for layer in cache_obj.layers:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                out.append((layer.keys, layer.values))
        if out:
            return out

    if isinstance(cache_obj, (list, tuple)):
        out = []
        for layer in cache_obj:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                out.append((layer.keys, layer.values))
            elif isinstance(layer, (list, tuple)) and len(layer) >= 2:
                out.append((layer[0], layer[1]))
        return out

    raise TypeError(f"Unsupported cache object type: {type(cache_obj)}")


def _to_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


@dataclass
class CacheLayerMetrics:
    layer_index: int
    raw_bytes: int
    compressed_bytes: int
    score_cosine: float
    top1_match: float
    top5_match: float


@dataclass
class CacheAnalysisSummary:
    mode: str
    bits_k: int
    bits_v: int
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    score_cosine_mean: float | None
    top1_match_mean: float | None
    top5_match_mean: float | None
    layers_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "bits_k": self.bits_k,
            "bits_v": self.bits_v,
            "raw_bytes": self.raw_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "score_cosine_mean": self.score_cosine_mean,
            "top1_match_mean": self.top1_match_mean,
            "top5_match_mean": self.top5_match_mean,
            "layers_evaluated": self.layers_evaluated,
        }


class TurboQuantCacheAnalyzer:
    """Compresses real HF cache tensors and reports size/fidelity statistics."""

    def __init__(
        self,
        bits_k: int = 3,
        bits_v: int = 2,
        seed: int = 42,
        score_backend: str = "xla",
        tile_size: int = 256,
        query_tile_size: int = 128,
    ) -> None:
        self.bits_k = int(bits_k)
        self.bits_v = int(bits_v)
        self.seed = int(seed)
        self.score_backend = score_backend
        self.tile_size = int(tile_size)
        self.query_tile_size = int(query_tile_size)

    def _score_metrics(self, real_scores: np.ndarray, approx_scores: np.ndarray) -> tuple[float, float, float]:
        real = real_scores.reshape(real_scores.shape[0], real_scores.shape[1], -1)
        approx = approx_scores.reshape(approx_scores.shape[0], approx_scores.shape[1], -1)

        cos_vals: list[float] = []
        top1: list[float] = []
        top5: list[float] = []
        for head_idx in range(real.shape[1]):
            rs = real[0, head_idx]
            ts = approx[0, head_idx]
            denom = (np.linalg.norm(rs) * np.linalg.norm(ts)) + 1e-12
            cos_vals.append(float(np.dot(rs, ts) / denom))

            real_top1 = int(np.argmax(rs))
            tq_top1 = int(np.argmax(ts))
            top1.append(1.0 if real_top1 == tq_top1 else 0.0)

            tq_top5 = np.argsort(ts)[-5:]
            top5.append(1.0 if real_top1 in tq_top5 else 0.0)

        return float(np.mean(cos_vals)), float(np.mean(top1)), float(np.mean(top5))

    def _analyze_layer_turboquant(self, layer_idx: int, keys: Any, values: Any) -> CacheLayerMetrics:
        keys_np = _to_numpy(keys).astype(np.float16, copy=False)
        values_np = _to_numpy(values).astype(np.float16, copy=False)

        keys_j = jnp.asarray(keys_np)
        values_j = jnp.asarray(values_np)

        _, heads, seq_len, d_key = keys_j.shape
        _, _, _, d_val = values_j.shape

        backend = "pallas" if self.score_backend == "pallas" else "xla"
        key_compressor = TurboQuantCompressorV2JAX(
            head_dim=int(d_key),
            bits=self.bits_k,
            seed=self.seed + layer_idx * 1000,
            tile_size=self.tile_size,
            query_tile_size=self.query_tile_size,
            score_backend=backend,
        )
        value_compressor = TurboQuantCompressorMSEJAX(
            head_dim=int(d_val),
            bits=self.bits_v,
            seed=self.seed + layer_idx * 1000 + 503,
        )

        compressed_k = key_compressor.compress(keys_j)
        compressed_v = value_compressor.compress(values_j)

        query = keys_j[:, :, -1:, :]
        real_scores = jnp.matmul(query.astype(jnp.float32), jnp.swapaxes(keys_j.astype(jnp.float32), -2, -1))
        tq_scores = key_compressor.asymmetric_attention_scores(query, compressed_k)

        score_cos, top1, top5 = self._score_metrics(np.asarray(real_scores), np.asarray(tq_scores))

        raw_bytes = _tensor_bytes(keys) + _tensor_bytes(values)
        compressed_bytes = _tensor_bytes(compressed_k) + _tensor_bytes(compressed_v)

        return CacheLayerMetrics(
            layer_index=layer_idx,
            raw_bytes=raw_bytes,
            compressed_bytes=compressed_bytes,
            score_cosine=score_cos,
            top1_match=top1,
            top5_match=top5,
        )

    def _analyze_layer_mse(self, layer_idx: int, keys: Any, values: Any) -> CacheLayerMetrics:
        keys_np = _to_numpy(keys).astype(np.float16, copy=False)
        values_np = _to_numpy(values).astype(np.float16, copy=False)

        keys_j = jnp.asarray(keys_np)
        values_j = jnp.asarray(values_np)

        _, _, _, d_key = keys_j.shape
        _, _, _, d_val = values_j.shape

        key_compressor = TurboQuantCompressorMSEJAX(
            head_dim=int(d_key),
            bits=self.bits_k,
            seed=self.seed + layer_idx * 1000,
        )
        value_compressor = TurboQuantCompressorMSEJAX(
            head_dim=int(d_val),
            bits=self.bits_v,
            seed=self.seed + layer_idx * 1000 + 503,
        )

        compressed_k = key_compressor.compress(keys_j)
        compressed_v = value_compressor.compress(values_j)

        query = keys_j[:, :, -1:, :]
        real_scores = jnp.matmul(query.astype(jnp.float32), jnp.swapaxes(keys_j.astype(jnp.float32), -2, -1))
        recon_k = key_compressor.decompress(compressed_k)
        approx_scores = jnp.matmul(query.astype(jnp.float32), jnp.swapaxes(recon_k.astype(jnp.float32), -2, -1))

        score_cos, top1, top5 = self._score_metrics(np.asarray(real_scores), np.asarray(approx_scores))

        raw_bytes = _tensor_bytes(keys) + _tensor_bytes(values)
        compressed_bytes = _tensor_bytes(compressed_k) + _tensor_bytes(compressed_v)

        return CacheLayerMetrics(
            layer_index=layer_idx,
            raw_bytes=raw_bytes,
            compressed_bytes=compressed_bytes,
            score_cosine=score_cos,
            top1_match=top1,
            top5_match=top5,
        )

    def analyze_cache(
        self,
        cache_obj: Any,
        mode: str = "turboquant",
        max_layers: int | None = None,
    ) -> CacheAnalysisSummary:
        mode_norm = mode.lower().strip()
        if mode_norm not in {"baseline", "mse", "turboquant"}:
            raise ValueError(f"Unsupported analysis mode: {mode}")

        layers = extract_cache_layers(cache_obj)
        if max_layers is not None and max_layers > 0:
            layers = layers[: max_layers]

        if not layers:
            return CacheAnalysisSummary(
                mode=mode_norm,
                bits_k=self.bits_k,
                bits_v=self.bits_v,
                raw_bytes=0,
                compressed_bytes=0,
                compression_ratio=1.0,
                score_cosine_mean=None,
                top1_match_mean=None,
                top5_match_mean=None,
                layers_evaluated=0,
            )

        per_layer: list[CacheLayerMetrics] = []
        for idx, (keys, values) in enumerate(layers):
            if mode_norm == "baseline":
                raw = _tensor_bytes(keys) + _tensor_bytes(values)
                per_layer.append(
                    CacheLayerMetrics(
                        layer_index=idx,
                        raw_bytes=raw,
                        compressed_bytes=raw,
                        score_cosine=1.0,
                        top1_match=1.0,
                        top5_match=1.0,
                    )
                )
            elif mode_norm == "mse":
                per_layer.append(self._analyze_layer_mse(idx, keys, values))
            else:
                per_layer.append(self._analyze_layer_turboquant(idx, keys, values))

        raw_total = sum(m.raw_bytes for m in per_layer)
        compressed_total = sum(m.compressed_bytes for m in per_layer)
        ratio = float(raw_total / compressed_total) if compressed_total > 0 else 1.0

        return CacheAnalysisSummary(
            mode=mode_norm,
            bits_k=self.bits_k,
            bits_v=self.bits_v,
            raw_bytes=raw_total,
            compressed_bytes=compressed_total,
            compression_ratio=ratio,
            score_cosine_mean=float(np.mean([m.score_cosine for m in per_layer])) if per_layer else None,
            top1_match_mean=float(np.mean([m.top1_match for m in per_layer])) if per_layer else None,
            top5_match_mean=float(np.mean([m.top5_match for m in per_layer])) if per_layer else None,
            layers_evaluated=len(per_layer),
        )


def summarize_cache_bytes(layers: Iterable[tuple[Any, Any]]) -> int:
    total = 0
    for keys, values in layers:
        total += _tensor_bytes(keys)
        total += _tensor_bytes(values)
    return total

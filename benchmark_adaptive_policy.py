from __future__ import annotations

import argparse
import time
from typing import Any

import jax
import jax.numpy as jnp

from turboquant_jax import JAXTurboQuantKVCache


def _block_tree(value: Any) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _time_call(fn, *args, repeats: int = 8, warmup: int = 2):
    result = None
    for _ in range(warmup):
        result = fn(*args)
        _block_tree(result)

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        _block_tree(result)
        timings.append((time.perf_counter() - start) * 1000.0)

    return sum(timings) / len(timings), result


def _make_kv(key: jax.Array, b: int, h: int, s: int, d: int):
    k1, k2 = jax.random.split(key)
    keys = jax.random.normal(k1, (b, h, s, d), dtype=jnp.float16)
    values = jax.random.normal(k2, (b, h, s, d), dtype=jnp.float16)

    keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
    values = values / (jnp.linalg.norm(values, axis=-1, keepdims=True) + 1e-8)
    return keys, values


def _run_policy(
    policy: str,
    score_backend: str,
    tile_size: int,
    b: int,
    h: int,
    s: int,
    d: int,
    num_reuse_queries: int,
):
    key = jax.random.PRNGKey(0)
    keys, values = _make_kv(key, b, h, s, d)
    query = keys[:, :, -1:, :]

    cache = JAXTurboQuantKVCache(
        d_key=d,
        d_value=d,
        bits=3,
        score_cache_policy=policy,
        score_backend=score_backend,
        tile_size=tile_size,
        adaptive_promote_after=2,
        adaptive_hit_rate_threshold=0.3,
        adaptive_memory_budget_mb=512.0,
    )

    cache.append(keys, values)

    # Simulate repeated query reuse over same prefix cache.
    def score_once(q):
        return cache.attention_scores(q)

    score_ms, _ = _time_call(score_once, query, repeats=max(2, num_reuse_queries), warmup=1)
    mem = cache.memory_usage_bits()
    stats = cache.policy_stats()

    return {
        "policy": policy,
        "score_backend": cache.key_compressor.score_backend,
        "score_ms": score_ms,
        "compression_ratio": mem["compression_ratio"],
        "runtime_compression_ratio": mem["runtime_compression_ratio"],
        "prepared_share": mem["prepared_cache_share"],
        "promotion_events": stats["promotion_events"],
        "prepared_score_calls": stats["prepared_score_calls"],
        "packed_score_calls": stats["packed_score_calls"],
        "cache_hit_rate": stats["cache_hit_rate"],
        "avg_reuse_per_segment": stats["avg_reuse_per_segment"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark adaptive cache policy for TurboQuant JAX")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--score-backend", choices=["xla", "pallas"], default="xla")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--reuse-queries", type=int, default=8)
    args = parser.parse_args()

    devices = jax.devices(args.device)
    if not devices:
        raise RuntimeError(f"No JAX devices found for {args.device}")

    policies = ["packed", "adaptive", "prepared"]
    rows = []
    for policy in policies:
        row = _run_policy(
            policy=policy,
            score_backend=args.score_backend,
            tile_size=args.tile_size,
            b=1,
            h=args.kv_heads,
            s=args.seq_len,
            d=args.head_dim,
            num_reuse_queries=args.reuse_queries,
        )
        rows.append(row)

    print("| Policy | Backend | Score (ms) | Stored compression | Runtime compression | Prepared share | Promotions | Prepared calls | Packed calls | Cache hit rate | Avg reuse |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(
            f"| {r['policy']} | {r['score_backend']} | {r['score_ms']:.2f} | {r['compression_ratio']:.2f}x | "
            f"{r['runtime_compression_ratio']:.2f}x | {r['prepared_share']:.2f} | {r['promotion_events']:.0f} | "
            f"{r['prepared_score_calls']:.0f} | {r['packed_score_calls']:.0f} | {r['cache_hit_rate']:.2f} | {r['avg_reuse_per_segment']:.2f} |"
        )


if __name__ == "__main__":
    main()

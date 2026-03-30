from __future__ import annotations

import argparse
import time
from typing import Any

import jax
import jax.numpy as jnp

from turboquant_jax.compressors import TurboQuantCompressorV2JAX
from turboquant_jax.quantization_utils import pack_low_bit_values, unpack_low_bit_values


def _block_tree(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _time_call(fn, *args, warmup: int = 2, repeats: int = 8) -> float:
    out = None
    for _ in range(warmup):
        out = fn(*args)
        _block_tree(out)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn(*args)
        _block_tree(out)
        times.append((time.perf_counter() - start) * 1000.0)
    return sum(times) / max(len(times), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbench hot TurboQuant kernels")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--query-len", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--score-backend", choices=["xla", "pallas"], default="xla")
    args = parser.parse_args()

    devices = jax.devices(args.device)
    if not devices:
        raise RuntimeError(f"No JAX {args.device} device available")
    device = devices[0]

    key = jax.random.PRNGKey(0)
    key, k1, k2, k3 = jax.random.split(key, 4)

    values = jax.random.randint(k1, (args.seq_len * args.head_dim,), 0, 2 ** args.bits, dtype=jnp.uint8)
    keys = jax.random.normal(k2, (1, args.heads, args.seq_len, args.head_dim), dtype=jnp.float32)
    queries = jax.random.normal(k3, (1, args.heads, args.query_len, args.head_dim), dtype=jnp.float32)

    values = jax.device_put(values, device)
    keys = jax.device_put(keys, device)
    queries = jax.device_put(queries, device)

    compressor = TurboQuantCompressorV2JAX(
        head_dim=args.head_dim,
        bits=args.bits,
        seed=42,
        score_backend=args.score_backend,
    )

    pack_ms = _time_call(lambda x: pack_low_bit_values(x, args.bits), values)
    packed, shape = pack_low_bit_values(values, args.bits)
    _block_tree((packed, shape))

    unpack_ms = _time_call(lambda x: unpack_low_bit_values(x, args.bits, shape), packed)
    quant_ms = _time_call(compressor.compress, keys)

    compressed = compressor.compress(keys)
    _block_tree(compressed)

    score_ms = _time_call(compressor.asymmetric_attention_scores, queries, compressed)

    print("=" * 72)
    print("TurboQuant Hot Kernel Microbenchmark")
    print("=" * 72)
    print(f"device={device}")
    print(f"bits={args.bits} seq_len={args.seq_len} head_dim={args.head_dim} heads={args.heads} query_len={args.query_len}")
    print(f"pack_ms={pack_ms:.3f}")
    print(f"unpack_ms={unpack_ms:.3f}")
    print(f"quantize_ms={quant_ms:.3f}")
    print(f"score_ms={score_ms:.3f}")


if __name__ == "__main__":
    main()

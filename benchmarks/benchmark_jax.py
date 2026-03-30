from __future__ import annotations

import argparse
import os
import time
import sys
from typing import Any

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant_jax import (
    calibrate_qjl_scale_step,
    init_turboquant_prod,
    pairwise_inner_products,
    prod_inner_product,
    prod_quantize,
)


def _block_tree(value: Any) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _time_call(fn, *args, repeats: int = 10, warmup: int = 2):
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


def _make_unit_vectors(key: jax.Array, n: int, d: int) -> jnp.ndarray:
    x = jax.random.normal(key, (n, d), dtype=jnp.float32)
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX benchmark for TurboQuant core functions")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Device type to run on")
    parser.add_argument("--bits", type=int, default=3, help="Total quantization bits")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--num-vectors", type=int, default=4096, help="Number of vectors")
    parser.add_argument("--num-queries", type=int, default=64, help="Number of query vectors")
    parser.add_argument("--repeats", type=int, default=10, help="Timed repeats")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup rounds")
    parser.add_argument("--qjl-dim", type=int, default=0, help="QJL projection dimension (0 means dim)")
    args = parser.parse_args()

    devices = jax.devices(args.device)
    if not devices:
        raise RuntimeError(f"No JAX devices found for {args.device}")
    device = devices[0]

    key = jax.random.PRNGKey(0)
    key_x, key_y, key_q = jax.random.split(key, 3)

    x = _make_unit_vectors(key_x, args.num_vectors, args.dim)
    y = _make_unit_vectors(key_y, args.num_vectors, args.dim)
    queries = _make_unit_vectors(key_q, args.num_queries, args.dim)

    x = jax.device_put(x, device)
    y = jax.device_put(y, device)
    queries = jax.device_put(queries, device)

    qjl_dim = args.qjl_dim if args.qjl_dim > 0 else args.dim

    init_start = time.perf_counter()
    state = init_turboquant_prod(args.dim, args.bits, qjl_dim=qjl_dim, seed=42)
    init_ms = (time.perf_counter() - init_start) * 1000.0

    quantize_ms, compressed = _time_call(
        prod_quantize,
        state,
        x,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    inner_ms, inner_vals = _time_call(
        prod_inner_product,
        state,
        y,
        compressed,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    pairwise_ms, pairwise_vals = _time_call(
        pairwise_inner_products,
        state,
        queries,
        compressed,
        repeats=max(2, args.repeats // 2),
        warmup=args.warmup,
    )

    x_mse = state.mse_state.centroids[compressed.mse_indices] @ state.mse_state.Pi
    term1 = jnp.sum(y * x_mse, axis=-1)
    y_projected = y @ state.S.T
    qjl_ip = jnp.sum(y_projected * compressed.qjl_signs, axis=-1)
    target_inner = jnp.sum(y * x, axis=-1)

    scale0 = jnp.asarray(jnp.sqrt(jnp.pi / 2.0) / float(state.qjl_dim), dtype=jnp.float32)

    autodiff_ms, (scale1, grad_val) = _time_call(
        calibrate_qjl_scale_step,
        scale0,
        term1,
        compressed.residual_norm,
        qjl_ip,
        target_inner,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    print(f"JAX device: {device}")
    print(f"vectors={args.num_vectors}, dim={args.dim}, bits={args.bits}, qjl_dim={qjl_dim}")
    print(f"state init:             {init_ms:8.2f} ms")
    print(f"prod_quantize (jit):    {quantize_ms:8.2f} ms")
    print(f"prod_inner_product:     {inner_ms:8.2f} ms")
    print(f"pairwise inner (vmap):  {pairwise_ms:8.2f} ms")
    print(f"autodiff step (grad):   {autodiff_ms:8.2f} ms")
    print(f"mean(inner):            {float(jnp.mean(inner_vals)):+.6f}")
    print(f"pairwise shape:         {tuple(pairwise_vals.shape)}")
    print(f"scale update:           {float(scale0):.6f} -> {float(scale1):.6f}, grad={float(grad_val):+.6f}")


if __name__ == "__main__":
    main()

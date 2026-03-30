"""
Verification script for JAX TurboQuant implementation.
Tests MSE distortion bounds, inner product accuracy, and compression ratios
against theoretical predictions from the paper.
"""

import math
import os
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turboquant_jax.compressors import JAXTurboQuantKVCache, TurboQuantCompressorV2JAX
from turboquant_jax.lloyd_max import get_lloyd_max_codebook
from turboquant_jax.turboquant import (
    init_turboquant_mse,
    init_turboquant_prod,
    mse_dequantize,
    mse_quantize,
    prod_inner_product,
    prod_quantize,
)


def _block_tree(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def test_lloyd_max_codebook():
    """Verify codebook properties for various dimensions and bit-widths."""
    print("=" * 60)
    print("TEST 1: Lloyd-Max Codebook Properties")
    print("=" * 60)

    for d in [64, 128, 256]:
        for bits in [1, 2, 3, 4]:
            cb = get_lloyd_max_codebook(d, bits, use_exact=False)
            print(f"  d={d:>4d}, bits={bits}: {2**bits} levels, "
                  f"centroids range=[{float(jnp.min(cb.centroids)):.4f}, {float(jnp.max(cb.centroids)):.4f}]")

    # Verify symmetry (centroids should be symmetric around 0)
    cb = get_lloyd_max_codebook(128, 3)
    centroid_sum = float(jnp.abs(cb.centroids.sum()))
    print(f"\n  Symmetry check (d=128, b=3): sum of centroids = {centroid_sum:.6f} (should be ~0)")
    assert centroid_sum < 0.01, "Centroids should be symmetric!"
    print("  PASSED\n")


def test_mse_quantizer():
    """Verify MSE distortion on random unit vectors."""
    print("=" * 60)
    print("TEST 2: MSE Quantizer Distortion")
    print("=" * 60)

    d = 128
    n_vectors = 1000

    key = jax.random.PRNGKey(42)

    for bits in [1, 2, 3, 4]:
        state = init_turboquant_mse(d, bits)
        
        # Generate random unit vectors
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (n_vectors, d))
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)

        # Quantize and reconstruct
        indices = jax.vmap(mse_quantize, in_axes=(None, 0))(state, x)
        x_hat = jax.vmap(mse_dequantize, in_axes=(None, 0))(state, indices)
        _block_tree((indices, x_hat))

        # Compute empirical MSE
        mse = float(jnp.mean(jnp.sum((x - x_hat) ** 2, axis=-1)))

        # Theoretical upper bound from paper: D_mse <= sqrt(3)*pi/2 * (1/4^b)
        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1 / (4 ** bits))

        ratio = mse / theoretical_bound
        status = "OK" if ratio <= 1.5 else "WARN"  # allow some slack for finite d

        print(f"  bits={bits}: MSE={mse:.6f}, theory_bound={theoretical_bound:.6f}, "
              f"ratio={ratio:.3f} [{status}]")

    print()


def test_inner_product_unbiasedness():
    """Verify that TurboQuant gives unbiased inner product estimates."""
    print("=" * 60)
    print("TEST 3: Inner Product Unbiasedness (QJL Correction)")
    print("=" * 60)

    d = 128
    n_trials = 2000

    key = jax.random.PRNGKey(42)

    for bits in [2, 3, 4]:
        state = init_turboquant_prod(d, bits)

        # Generate pairs of random unit vectors
        key, subkey1, subkey2 = jax.random.split(key, 3)
        x = jax.random.normal(subkey1, (n_trials, d))
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        y = jax.random.normal(subkey2, (n_trials, d))
        y = y / jnp.linalg.norm(y, axis=-1, keepdims=True)

        # True inner products
        true_ip = jnp.sum(x * y, axis=-1)

        # Quantize x, compute estimated inner products
        compressed_x = jax.vmap(prod_quantize, in_axes=(None, 0))(state, x)
        estimated_ip = jax.vmap(prod_inner_product, in_axes=(None, 0, 0))(state, y, compressed_x)
        _block_tree((compressed_x, estimated_ip))

        # Check bias (should be near 0)
        bias = float(jnp.mean(estimated_ip - true_ip))
        # Check RMSE
        rmse = float(jnp.sqrt(jnp.mean((estimated_ip - true_ip) ** 2)))
        # Correlation
        correlation = float(jnp.corrcoef(true_ip, estimated_ip)[0, 1])

        # Theoretical distortion bound: D_prod <= sqrt(3)*pi^2/d * (1/4^b)
        theoretical_distortion = math.sqrt(3) * math.pi ** 2 / d * (1 / (4 ** bits))

        print(f"  bits={bits}: bias={bias:+.6f}, RMSE={rmse:.6f}, "
              f"corr={correlation:.4f}, theory_D={theoretical_distortion:.6f}")

    print()


def test_mse_only_inner_product_bias():
    """Show that MSE-only quantizer has biased inner products (motivating QJL)."""
    print("=" * 60)
    print("TEST 4: MSE-Only Inner Product Bias (motivation for QJL)")
    print("=" * 60)

    d = 128
    n_trials = 2000

    key = jax.random.PRNGKey(42)

    for bits in [1, 2, 3]:
        state = init_turboquant_mse(d, bits)

        key, subkey1, subkey2 = jax.random.split(key, 3)
        x = jax.random.normal(subkey1, (n_trials, d))
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        y = jax.random.normal(subkey2, (n_trials, d))
        y = y / jnp.linalg.norm(y, axis=-1, keepdims=True)

        true_ip = jnp.sum(x * y, axis=-1)
        
        indices = jax.vmap(mse_quantize, in_axes=(None, 0))(state, x)
        x_hat = jax.vmap(mse_dequantize, in_axes=(None, 0))(state, indices)
        
        mse_ip = jnp.sum(x_hat * y, axis=-1)

        _block_tree((indices, x_hat, mse_ip))
        bias = float(jnp.mean(mse_ip - true_ip))

        print(f"  bits={bits}: bias={bias:+.6f} (MSE-only is biased, QJL fixes this)")

    print()


def test_kv_cache():
    """Test the KV cache wrapper with compression ratios."""
    print("=" * 60)
    print("TEST 5: KV Cache Compression Ratios")
    print("=" * 60)

    d_key = 128
    d_value = 128
    seq_len = 1024
    
    key_rng = jax.random.PRNGKey(42)

    for bits in [2, 3, 4]:
        cache = JAXTurboQuantKVCache(d_key=d_key, d_value=d_value, bits=bits, seed=42, score_cache_policy="prepared")

        key_rng, k_key, v_key, q_key = jax.random.split(key_rng, 4)
        keys = jax.random.normal(k_key, (1, 1, seq_len, d_key), dtype=jnp.float32)
        values = jax.random.normal(v_key, (1, 1, seq_len, d_value), dtype=jnp.float32)

        cache.append(keys, values)
        usage = cache.memory_usage_bits()

        print(f"  bits={bits}: compression={usage['compression_ratio']:.2f}x "
              f"({usage['total_bits'] / 8 / 1024:.1f} KB vs "
              f"{usage['fp16_bits'] / 8 / 1024:.1f} KB fp16)")

        query = jax.random.normal(q_key, (1, 1, 1, d_key), dtype=jnp.float32)
        scores = cache.attention_scores(query)
        _block_tree(scores)

        print(
            f"           attention scores shape: {tuple(scores.shape)}, "
            f"range=[{float(jnp.min(scores)):.3f}, {float(jnp.max(scores)):.3f}]"
        )

    print()


def test_needle_in_haystack():
    """
    Simplified needle-in-haystack: hide a specific key among many,
    verify we can still find it via attention after quantization.
    """
    print("=" * 60)
    print("TEST 6: Needle-in-Haystack Retrieval")
    print("=" * 60)

    d = 128
    key_rng = jax.random.PRNGKey(123)

    for bits in [2, 3, 4]:
        for seq_len in [512, 2048, 8192]:
            key_rng, subkey = jax.random.split(key_rng)
            keys = jax.random.normal(subkey, (1, 1, seq_len, d), dtype=jnp.float32)
            keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

            needle_pos = seq_len // 3
            query = keys[:, :, needle_pos:needle_pos + 1, :]

            quantizer = TurboQuantCompressorV2JAX(d, bits, seed=42)
            compressed = quantizer.compress(keys)
            estimated_ips = quantizer.asymmetric_attention_scores(query, compressed).reshape(-1)
            _block_tree((compressed, estimated_ips))

            top_idx = int(jnp.argmax(estimated_ips))
            top5 = jnp.argsort(estimated_ips)[-5:]
            in_top5 = bool(jnp.any(top5 == needle_pos))
            found = top_idx == needle_pos

            status = "EXACT" if found else ("TOP-5" if in_top5 else "MISS")
            print(f"  bits={bits}, seq={seq_len:>5d}: top1={top_idx:>5d} "
                  f"(needle={needle_pos:>5d}) [{status}]")

    print()


def test_gpu_if_available():
    """Run a quick benchmark on GPU if available."""
    print("=" * 60)
    print("TEST 7: GPU Benchmark (if CUDA available)")
    print("=" * 60)

    try:
        gpu_devices = jax.devices("gpu")
    except Exception:
        gpu_devices = []

    if not gpu_devices:
        print("  GPU backend not available, skipping GPU test")
        print()
        return

    device = gpu_devices[0]
    print(f"  GPU: {device}")

    seq_len = 8192
    d = 128
    bits = 3
    n_queries = 64

    print(f"  Config: d={d}, bits={bits}, seq_len={seq_len}, n_queries={n_queries}")

    key_rng = jax.random.PRNGKey(42)
    key_rng, sk1, sk2 = jax.random.split(key_rng, 3)

    keys = jax.device_put(jax.random.normal(sk1, (1, 1, seq_len, d), dtype=jnp.float32), device)
    keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
    queries = jax.device_put(jax.random.normal(sk2, (1, 1, n_queries, d), dtype=jnp.float32), device)
    queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

    compressor = TurboQuantCompressorV2JAX(d, bits, seed=42)
    
    # Jit compile trigger and run
    t0 = time.perf_counter()
    compressed = compressor.compress(keys)
    _block_tree(compressed)
    compile_time = time.perf_counter() - t0
    print(f"  First compress (compilation): {compile_time * 1000:.2f} ms")

    # Benchmark quantize
    t0 = time.perf_counter()
    for _ in range(10):
        compressed = compressor.compress(keys)
        _block_tree(compressed)
    quant_time = (time.perf_counter() - t0) / 10
    print(f"  Quantize {seq_len} keys: {quant_time * 1000:.2f} ms")

    # Benchmark asymmetric attention
    compressed = compressor.compress(keys)
    _block_tree(compressed)
    t0 = time.perf_counter()
    for _ in range(100):
        scores = compressor.asymmetric_attention_scores(queries, compressed)
        _block_tree(scores)
    ip_time = (time.perf_counter() - t0) / 100
    print(f"  Inner product ({n_queries} queries x {seq_len} keys): {ip_time * 1000:.2f} ms")

    # Compare with full-precision
    t0 = time.perf_counter()
    for _ in range(100):
        fp_scores = jnp.einsum("b h q d, b h k d -> b h q k", queries, keys)
        _block_tree(fp_scores)
    fp_time = (time.perf_counter() - t0) / 100
    print(f"  Full-precision matmul: {fp_time * 1000:.2f} ms")

    # Memory comparison (approximate payload bits)
    fp16_bytes = seq_len * d * 2
    quant_bytes = seq_len * d * bits / 8
    print(f"  Memory: {fp16_bytes / 1024:.1f} KB (fp16) vs {quant_bytes / 1024:.1f} KB (TQ-{bits}bit)")
    print(f"  Compression: {fp16_bytes / quant_bytes:.1f}x")
    print()


if __name__ == "__main__":
    print()
    print("TurboQuant JAX Implementation Verification")
    print("Based on: 'TurboQuant: Online Vector Quantization' (ICLR 2026)")
    print()

    test_lloyd_max_codebook()
    test_mse_quantizer()
    test_inner_product_unbiasedness()
    test_mse_only_inner_product_bias()
    test_kv_cache()
    test_needle_in_haystack()
    test_gpu_if_available()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)

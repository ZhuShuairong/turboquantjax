# Benchmark Results Snapshot

Last full rerun: 2026-03-28
Environment: WSL Ubuntu, conda env turboquant-jax, NVIDIA GPU

## TurboQuant JAX Qwen3.5 Comparison

Settings:
- Device: cuda:0
- Synthetic KV sequence length: 2048
- Score cache policy: prepared
- Key tile size: 256
- Query tile size: 128
- Fused score backend: xla

### Method Summary

| Method | Avg baseline score (ms) | Avg compressed score (ms) | Avg compression |
| --- | ---: | ---: | ---: |
| baseline-fp16 | 1.25 | 1.25 | 1.00x |
| mse-only-jax-packed-2bit | 1.25 | 3.01 | 7.53x |
| mse-only-jax-packed-3bit | 1.25 | 2.11 | 5.12x |
| mse-only-jax-packed-4bit | 1.25 | 2.06 | 3.88x |
| turboquant-jax-packed-2bit | 1.25 | 1.33 | 7.31x |
| turboquant-jax-packed-3bit | 1.25 | 1.38 | 5.02x |
| turboquant-jax-packed-4bit | 1.25 | 1.40 | 3.82x |

## TurboQuant JAX Adaptive Policy Comparison

Settings:
- Device: GPU (WSL)
- Backend: xla
- Sequence length: 512
- Tile sizes: key=128, query=128
- Reuse queries: 8

| Policy | Backend | Score (ms) | Stored compression | Runtime compression | Prepared share | Promotions | Prepared calls | Packed calls | Cache hit rate | Avg reuse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| packed | xla | 1.26 | 5.02x | 5.02x | 0.00 | 0 | 0 | 9 | 1.00 | 9.00 |
| adaptive | xla | 126.86 | 5.02x | 1.41x | 0.72 | 1 | 8 | 1 | 1.00 | 9.00 |
| prepared | xla | 1.18 | 5.02x | 1.41x | 0.72 | 0 | 9 | 0 | 1.00 | 9.00 |

Readout:
- prepared gives best repeated-query latency in this workload.
- packed gives best runtime memory compression.
- adaptive promotes quickly but has transition overhead in short bursty reuse.

## WSL Fair Base vs GGUF Inference Comparison

Settings:
- Decode tokens: 24
- Context tokens: 256, 512, 1024
- GGUF n_gpu_layers: -1

### Context Scaling Summary

| Model | Family | Prefill tok/s @ min context | Prefill tok/s @ max context | Prefill scaling | Decode tok/s @ min context | Decode tok/s @ max context |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | base-fp16 | 8.6 | 10.7 | 1.24x | 8.2 | 7.8 |
| Qwen3.5-4B | gguf-iq4_xs(layers=-1) | 9965.9 | 5693.1 | 0.57x | 61.4 | 59.5 |
| Qwen3.5-9B | base-fp16 | 8.3 | 8.5 | 1.02x | 0.8 | 0.8 |
| Qwen3.5-9B | gguf-iq4_xs(layers=-1) | 7430.8 | 3872.1 | 0.52x | 45.8 | 40.8 |

## Reproduce

Run these in WSL from the turboquant-jax folder:

- python benchmark_jax_qwen35.py --device gpu --bits 2 3 4 --seq-len 2048 --score-cache-policy prepared --score-backend xla --tile-size 256 --query-tile-size 128
- python benchmark_adaptive_policy.py --device gpu --score-backend xla --tile-size 128 --query-tile-size 128 --seq-len 512 --reuse-queries 8
- python /mnt/c/Users/zshua/Downloads/TQ-Experimentation/turboquant-pytorch/benchmark_wsl_base_vs_gguf.py --contexts 256 512 1024 --decode-tokens 24

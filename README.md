# TurboQuant JAX

JAX implementation of TurboQuant core math with:
- JIT compilation for quantize and inner product paths
- VMAP for batched query scoring
- AutoDiff for calibration of the QJL correction scale
- Packed storage compressors and a KV cache wrapper API

## Project Layout

- `turboquant_jax/lloyd_max.py`: Lloyd-Max scalar codebook solver with cache
- `turboquant_jax/quantization_utils.py`: Rotation, QJL matrix generation, and bit packing helpers
- `turboquant_jax/turboquant.py`: TurboQuant MSE and Prod core math
- `turboquant_jax/fused_kernels.py`: Fused dequantization plus attention term1 kernels (XLA and optional Pallas)
- `turboquant_jax/compressors.py`: Packed key and value compressors plus `JAXTurboQuantKVCache` with adaptive policy
- `benchmark_jax.py`: Microbenchmark for JIT, VMAP, and AutoDiff paths
- `benchmark_jax_qwen35.py`: GPU benchmark across local Qwen3.5 base configs with markdown report generation
- `benchmark_adaptive_policy.py`: Packed versus adaptive versus prepared policy benchmark with reuse statistics
- `scripts/run_wsl_benchmark.ps1`: PowerShell helper that syncs project into WSL and runs benchmarks in conda

## WSL Setup (Recommended)

These steps create a WSL environment that can run JAX on NVIDIA GPU.

### 1. Install WSL and Ubuntu

Run in elevated PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then open Ubuntu and finish first time user setup.

### 2. Verify GPU inside WSL

In Ubuntu:

```bash
nvidia-smi
```

If this fails, update Windows NVIDIA drivers first.

### 3. Install Miniconda in WSL

In Ubuntu:

```bash
mkdir -p ~/tmp && cd ~/tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
```

### 4. Clone the repository and create env

In Ubuntu:

```bash
git clone https://github.com/ZhuShuairong/turboquantjax.git
cd turboquantjax
conda env create -f environment.yml
conda activate turboquant-jax
```

### 5. Install CUDA JAX wheel

In Ubuntu with the env activated:

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda12]"
```

### 6. Verify JAX device

```bash
python -c "import jax; print(jax.devices())"
```

Expected output includes a CUDA device.

### 7. Run benchmarks

```bash
python benchmark_jax.py --device gpu --bits 3 --dim 128 --num-vectors 4096
python benchmark_jax_qwen35.py --device gpu --bits 2 3 4 --seq-len 2048 --score-cache-policy prepared --score-backend xla --tile-size 256 --query-tile-size 128
python benchmark_adaptive_policy.py --device gpu --score-backend xla --tile-size 128 --query-tile-size 128 --seq-len 512 --reuse-queries 8
```

## Next Pass Optimizations

This repository now includes the next optimization pass for the remaining bottlenecks.

### 1. Device-side packing and unpacking

- Bit packing and unpacking are now fully implemented with JAX arrays.
- The previous NumPy host loop path has been removed from the hot path.

### 2. Fused dequantization plus attention matmul path

- The tiled scoring path rotates queries once (`q @ Pi^T`) and computes term1 directly from codebook indices.
- This avoids reconstructing full dequantized key tensors before matmul.
- Packed scoring now decodes only the active key tile from packed bitstreams, instead of unpacking the entire sequence on each score call.
- Scoring uses two-dimensional tiling (key tiles and query tiles) to reduce temporary tensor size and improve batching behavior.
- Backend selection:
	- `xla` (default, stable)
	- `pallas` (optional, automatically falls back to `xla` if unsupported in the current JAX runtime)

### 3. Adaptive runtime policy switch

`JAXTurboQuantKVCache` supports three score cache policies:

- `packed`: minimum runtime memory, higher repeated-query latency
- `prepared`: lower repeated-query latency, higher runtime memory due to prepared tensors
- `adaptive`: promotes prefixes from packed to prepared using hit-rate and reuse statistics

Adaptive policy controls:

- `adaptive_promote_after`: minimum reuse count before promotion
- `adaptive_hit_rate_threshold`: required cache-hit rate before promotion is allowed
- `adaptive_memory_budget_mb`: optional cap on prepared-cache memory

Runtime statistics are exposed by `policy_stats()` and include:

- cache hit rate
- prepared versus packed score calls
- promotion events
- average reuse per cached segment

## Production Guidance

- Choose `packed` for memory-constrained, low-reuse traffic.
- Choose `prepared` for high-reuse prefixes where latency is critical.
- Choose `adaptive` for mixed traffic where reuse varies over time.
- Tune `tile_size` by device and model shape, typically 128 or 256 as starting points.

## Optional Windows Helper Script

From Windows PowerShell in this folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_wsl_benchmark.ps1
```

This script copies files into `~/TQ-Experimentation/turboquant-jax` in WSL, activates conda env `turboquant-jax`, and runs both benchmark scripts.

## Findings (Markdown)

Benchmarks were run on Qwen3.5 base model configs with GPU-enabled JAX in WSL.
Latest full rerun: 2026-03-28.
Latest all-model safe rerun: 2026-03-28.

### All-Model KV Cache Comparison (Safe Mode)

To avoid desktop/display instability on 8 GB VRAM GPUs, use the safe-mode profile:

```bash
python benchmark_turboquant_vs_llamacpp_kv.py \
	--all-models \
	--gguf-root /mnt/c/models/gguf \
	--model-include qwen \
	--model-exclude mmproj \
	--contexts 2048 4096 8192 \
	--decode-tokens 6 \
	--llama-cache-types f16 q8_0 q4_0 \
	--llama-n-gpu-layers -1 \
	--llama-threads 6 \
	--llama-threads-batch 6 \
	--llama-n-batch 256 \
	--turboquant-bits 2 3 4 \
	--turboquant-policies packed prepared \
	--device gpu \
	--score-backend xla \
	--tile-size 256 \
	--query-tile-size 128 \
	--vram-budget-mb 8192 \
	--cooldown-s 2.0 \
	--safe-mode \
	--safe-max-context 8192 \
	--safe-decode-tokens 4 \
	--safe-llama-n-gpu-layers 16 \
	--safe-llama-threads 4 \
	--safe-llama-threads-batch 4 \
	--safe-llama-n-batch 128 \
	--safe-turboquant-bits 2 3 \
	--safe-turboquant-policies packed \
	--report-path /mnt/c/Users/zshua/Downloads/TQ-Experimentation/benchmark_qwen35_turboquant_rotorquant.md \
	--json-output /mnt/c/Users/zshua/Downloads/TQ-Experimentation/kv_compare_all_models_safe.json
```

Safe-mode run profile used in the latest all-model benchmark:

- Models benchmarked: `Qwen3.5-4B-IQ4_XS`, `Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M`, `Qwen3.5-9B-IQ4_XS`
- Context sweep: `2048, 4096, 8192`
- Decode tokens: `4`
- llama.cpp offload cap: `n_gpu_layers=16`
- TurboQuant matrix in safe mode: packed policies with `2/3-bit`

Observed outcome at matched context (8192 tokens):

- llama.cpp f16 KV estimate: `1024 MB`
- TurboQuant packed 2-bit runtime KV: `134 MB`
- KV memory improvement: `7.64x`

Note on current llama.cpp runtime:

- `q8_0` and `q4_0` KV cache modes failed with `Failed to create llama_context` in this environment.
- f16 KV mode completed successfully and is the baseline used in the matched-context comparison.

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

### Adaptive Policy Summary

| Policy | Score (ms) | Stored compression | Runtime compression |
| --- | ---: | ---: | ---: |
| packed | 1.26 | 5.02x | 5.02x |
| adaptive | 126.86 | 5.02x | 1.41x |
| prepared | 1.18 | 5.02x | 1.41x |

### Observations

- TurboQuant JAX remains much closer to baseline score latency than MSE-only at the same bit-width.
- 2-bit settings provide the highest stored compression, with expected quality and runtime tradeoffs.
- Packed policy maximizes runtime memory compression; prepared policy minimizes repeated-query latency.
- Adaptive policy can incur transition overhead under short, bursty reuse patterns.

## Notes

- This project currently focuses on quantization primitives and benchmark workflows.
- It does not yet patch compressed KV cache directly into full end-to-end generation loops.
- In-repo benchmark snapshot is available in `BENCHMARK_RESULTS.md`.

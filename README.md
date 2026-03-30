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
- `TURBOQUANT_APPLY_PLAYBOOK.md`: AI-oriented implementation playbook for applying TurboQuant KV compression to GGUF and base-model runtimes
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

## Scripts

### test_turboquant.py - Synthetic Validation

Validates the core algorithm without requiring a model download.

What it checks:
- Lloyd-Max codebook symmetry and ranges across dimensions and bit-widths
- MSE distortion against theoretical bounds
- Inner-product bias and correlation under QJL correction
- MSE-only bias (motivation for QJL)
- KV cache compression ratio and attention score path
- Needle-in-haystack retrieval behavior
- Optional GPU speed benchmark

Run in WSL directly:

```bash
python3 test_turboquant.py
```

Run from Windows PowerShell via helper:

```powershell
powershell -ExecutionPolicy Bypass -File run_tests_wsl.ps1
```

### validate.py - Real Model Validation

Runs real-model KV cache validation on Qwen2.5-3B-Instruct and compares
TurboQuant compressed attention scores with full-precision scores.

What it reports:
- Compression ratio per bit-width
- Attention score cosine similarity
- Top-1 and Top-5 match rates
- Needle token rank behavior

Run in WSL directly:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python3 validate.py
```

Run from Windows PowerShell via helper:

```powershell
powershell -ExecutionPolicy Bypass -File run_validate_wsl.ps1
```

## Optional Windows Helper Script

From Windows PowerShell in this folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_wsl_benchmark.ps1
```

This script copies files into `~/TQ-Experimentation/turboquant-jax` in WSL, activates conda env `turboquant-jax`, and runs both benchmark scripts.

## Findings (Markdown)

Benchmarks were run in WSL with GPU-enabled JAX and llama.cpp.
Latest all-model safe rerun: 2026-03-28.

### Important Comparison Tables

These tables focus only on the key research question: default llama.cpp KV versus TurboQuant KV for context capacity.

#### Matched Context (8192 tokens)

| Model | llama.cpp default KV (MB) | TurboQuant KV (MB) | KV reduction | Projected max context @ 8GB (llama.cpp) | Projected max context @ 8GB (TurboQuant) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B-IQ4_XS | 1024.0 | 134.0 | 7.64x | 65536 | 500812 |
| Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M | 1024.0 | 134.0 | 7.64x | 65536 | 500812 |
| Qwen3.5-9B-IQ4_XS | 1024.0 | 134.0 | 7.64x | 65536 | 500812 |

#### Context Scaling Example (Qwen3.5-4B-IQ4_XS)

| Context tokens | llama.cpp default KV (MB) | TurboQuant KV (MB) | KV reduction |
| ---: | ---: | ---: | ---: |
| 2048 | 256.0 | 33.5 | 7.64x |
| 4096 | 512.0 | 67.0 | 7.64x |
| 8192 | 1024.0 | 134.0 | 7.64x |

#### Throughput Snapshot at 8192 tokens

| Model | llama.cpp prefill (tok/s) | llama.cpp decode (tok/s) | TurboQuant score kernel (tok/s) |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B-IQ4_XS | 730.9 | 18.8 | 11.3 |
| Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M | 630.2 | 14.4 | 12.0 |
| Qwen3.5-9B-IQ4_XS | 400.4 | 10.3 | 10.6 |

#### llama.cpp KV Type Support in This Runtime

| Model | q8_0 KV mode | q4_0 KV mode |
| --- | --- | --- |
| Qwen3.5-4B-IQ4_XS | load-failed (`Failed to create llama_context`) | load-failed (`Failed to create llama_context`) |
| Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M | load-failed (`Failed to create llama_context`) | load-failed (`Failed to create llama_context`) |
| Qwen3.5-9B-IQ4_XS | load-failed (`Failed to create llama_context`) | load-failed (`Failed to create llama_context`) |

### Bottom Line

- In this setup, TurboQuant packed 2-bit KV reduced runtime KV memory by 7.64x versus default llama.cpp f16 KV at matched context.
- That reduction translates to higher projected context capacity under the same 8GB KV budget.
- Full concise benchmark report is in `benchmark_qwen35_turboquant_rotorquant.md`.

## Notes

- This project currently focuses on quantization primitives and benchmark workflows.
- It does not yet patch compressed KV cache directly into full end-to-end generation loops.
- In-repo benchmark snapshot is available in `BENCHMARK_RESULTS.md`.

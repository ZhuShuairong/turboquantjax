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
- `turboquant_jax/compressors.py`: Packed key and value compressors plus `JAXTurboQuantKVCache`
- `benchmark_jax.py`: Microbenchmark for JIT, VMAP, and AutoDiff paths
- `benchmark_jax_qwen35.py`: GPU benchmark across local Qwen3.5 base configs with markdown report generation
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
python benchmark_jax_qwen35.py --device gpu --bits 2 3 4 --seq-len 2048
```

## Optional Windows Helper Script

From Windows PowerShell in this folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_wsl_benchmark.ps1
```

This script copies files into `~/TQ-Experimentation/turboquant-jax` in WSL, activates conda env `turboquant-jax`, and runs both benchmark scripts.

## Findings (Markdown)

Benchmarks were run on Qwen3.5 base model configs with GPU-enabled JAX in WSL.

### Method Summary

| Method | Avg baseline score (ms) | Avg compressed score (ms) | Avg compression |
| --- | ---: | ---: | ---: |
| baseline-fp16 | 1.08 | 1.08 | 1.00x |
| mse-only-jax-packed-2bit | 1.08 | 1425.50 | 7.53x |
| mse-only-jax-packed-3bit | 1.08 | 1986.89 | 5.12x |
| mse-only-jax-packed-4bit | 1.08 | 2396.56 | 3.88x |
| turboquant-jax-packed-2bit | 1.08 | 1736.14 | 7.31x |
| turboquant-jax-packed-3bit | 1.08 | 2202.37 | 5.02x |
| turboquant-jax-packed-4bit | 1.08 | 2724.39 | 3.82x |

### Observations

- Packed JAX methods provide strong KV storage reduction.
- 2-bit packing shows the best compression ratio.
- Baseline fp16 scoring remains much faster in this version.
- Compression and compressed scoring are dominated by packing and unpacking overhead in Python level logic.

## Notes

- This project currently focuses on quantization primitives and benchmark workflows.
- It does not yet patch compressed KV cache directly into full end to end generation loops.

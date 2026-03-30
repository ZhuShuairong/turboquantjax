# Quickstart (Windows 11 + WSL2)

## 1) WSL + GPU baseline checks (PowerShell)

```powershell
wsl --install -d Ubuntu
powershell -ExecutionPolicy Bypass -File scripts/run_wsl_setup_check.ps1
```

## 2) Set up inside WSL (Ubuntu)

```bash
cd /mnt/c/Users/<you>/Downloads/TQ-Experimentation/turboquant-jax
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -e .[hf]
# Optional GGUF runtime
pip install -e .[gguf]
```

## 3) Environment diagnostics

```bash
python scripts/env_check.py
```

## 4) HF runtime test

```bash
python scripts/run_hf_qwen35.py \
  --model Qwen/Qwen3.5-2B \
  --cache turboquant \
  --max-new-tokens 64
```

## 5) GGUF runtime test

```bash
python scripts/run_gguf_qwen35.py \
  --model-path /mnt/c/models/gguf/Qwen3.5-9B-Q4_K_M.gguf \
  --runtime llamacpp-python \
  --cache baseline \
  --max-new-tokens 64
```

## 6) Benchmark matrix

```bash
python scripts/benchmark_matrix.py \
  --hf-model Qwen/Qwen3.5-2B \
  --gguf-model-path /mnt/c/models/gguf/Qwen3.5-9B-Q4_K_M.gguf \
  --contexts 2048,4096,8192,16384
```

## 7) Validation

```bash
python scripts/validate_against_baseline.py \
  --backend hf \
  --model Qwen/Qwen3.5-2B \
  --cache turboquant
```

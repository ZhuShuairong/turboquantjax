# Quickstart (Linux)

## 1) Prerequisites

- Linux x86_64
- Python 3.10+
- Optional NVIDIA GPU with CUDA driver
- Optional llama.cpp binaries (`llama-cli`, `llama-server`) for GGUF runtime

## 2) Install

```bash
cd turboquant-jax
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -e .[hf]
# Optional GGUF runtime
pip install -e .[gguf]
```

## 3) Validate Environment

```bash
python scripts/env_check.py
```

## 4) HF Inference

```bash
tqjax-generate \
  --backend hf \
  --model Qwen/Qwen3.5-2B \
  --cache turboquant \
  --bits-k 3 \
  --bits-v 2 \
  --prompt "Explain why KV caching matters for decode speed." \
  --max-new-tokens 64
```

## 5) GGUF Inference

```bash
tqjax-generate \
  --backend gguf \
  --model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --runtime llamacpp-python \
  --cache baseline \
  --prompt "Summarize this report in 6 bullets." \
  --max-new-tokens 64
```

If you run llama-server separately:

```bash
tqjax-generate \
  --backend gguf \
  --model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --runtime llama-server \
  --server-url http://127.0.0.1:8080 \
  --cache baseline \
  --prompt "Give me a short summary."
```

## 6) Benchmark

```bash
tqjax-bench \
  --backend hf \
  --model Qwen/Qwen3.5-2B \
  --contexts 2048,4096,8192,16384,32768 \
  --cache-modes baseline,mse,turboquant \
  --export-json artifacts/hf_bench.json \
  --export-md artifacts/hf_bench.md
```

## 7) Validate

```bash
tqjax-validate \
  --backend hf \
  --model Qwen/Qwen3.5-2B \
  --cache turboquant \
  --max-new-tokens 48 \
  --export-json artifacts/validate_hf.json \
  --export-md artifacts/validate_hf.md
```

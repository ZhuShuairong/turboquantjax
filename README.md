# TurboQuant JAX: Local Qwen3.5 Inference Runtime

TurboQuant JAX now provides a practical dual-backend local inference stack for Qwen-family models:

- Hugging Face backend for Qwen3.5 base/instruct checkpoints
- GGUF backend through llama.cpp runtimes (python binding, CLI, or llama-server)
- TurboQuant KV compression analysis integrated into generation and benchmark workflows

The original research-faithful TurboQuant math implementation remains intact, and this repo now layers a production CLI/API on top.

## Highlights

- Unified runtime package: `turboquant_jax/runtime`
- End-user CLIs:
  - `tqjax-generate`
  - `tqjax-bench`
  - `tqjax-serve`
  - `tqjax-validate`
  - `tqjax-env`
- Linux and Windows+WSL scripts under `scripts/`
- Expanded tests under `tests/` including unit, numerical, and integration smoke coverage

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[hf]
pip install -e .[gguf]
pip install -e .[dev]
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

## Environment Checks

```bash
python scripts/env_check.py
```

Windows PowerShell -> WSL diagnostics:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_wsl_setup_check.ps1
```

## Quick Usage

### HF generation

```bash
tqjax-generate \
  --backend hf \
  --model Qwen/Qwen3.5-2B \
  --cache turboquant \
  --bits-k 3 \
  --bits-v 2 \
  --prompt "Explain TurboQuant KV compression in practical terms." \
  --max-new-tokens 64
```

### GGUF generation

```bash
tqjax-generate \
  --backend gguf \
  --model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --runtime llamacpp-python \
  --cache baseline \
  --prompt "Summarize this note in five bullets." \
  --max-new-tokens 64
```

### HF benchmark matrix

```bash
tqjax-bench \
  --backend hf \
  --model Qwen/Qwen3.5-2B \
  --contexts 2048,4096,8192,16384,32768 \
  --cache-modes baseline,mse,turboquant \
  --export-json artifacts/hf_bench.json \
  --export-md artifacts/hf_bench.md
```

### GGUF benchmark matrix

```bash
tqjax-bench \
  --backend gguf \
  --model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --runtime llamacpp-python \
  --contexts 2048,4096,8192,16384,32768 \
  --cache-modes baseline,turboquant \
  --export-json artifacts/gguf_bench.json \
  --export-md artifacts/gguf_bench.md
```

### Validation workflow

```bash
tqjax-validate \
  --backend hf \
  --model Qwen/Qwen3.5-2B \
  --cache turboquant \
  --max-new-tokens 48 \
  --export-json artifacts/validate_hf.json \
  --export-md artifacts/validate_hf.md
```

### Serving workflow (GGUF)

```bash
tqjax-serve \
  --backend gguf \
  --model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --context-length 32768
```

## Script Wrappers

- `python scripts/run_hf_qwen35.py --model ...`
- `python scripts/run_gguf_qwen35.py --model-path ...`
- `python scripts/benchmark_hf_qwen35.py --model ...`
- `python scripts/benchmark_gguf_qwen35.py --model-path ...`
- `python scripts/benchmark_matrix.py --hf-model ... --gguf-model-path ...`
- `python scripts/validate_against_baseline.py --backend ...`

## Test Commands

Fast local suite:

```bash
pytest -q
```

Integration smoke tests (requires local model/runtime setup):

```bash
TQJAX_TEST_HF_MODEL=Qwen/Qwen3.5-2B pytest -q -m integration tests/test_hf_inference_smoke.py
TQJAX_TEST_GGUF_MODEL=/models/Qwen3.5-9B-Q4_K_M.gguf pytest -q -m integration tests/test_gguf_inference_smoke.py
```

## Runtime Modes and Scope

### HF backend cache modes

- `baseline`: native cache path
- `mse`: MSE-only compressed cache analysis
- `turboquant`: TurboQuant compressed cache analysis with score-fidelity statistics

### GGUF backend cache modes

- `baseline`: production local inference through llama.cpp runtime path
- `mse` and `turboquant`: staged integration in this milestone
  - generation is executed by robust llama.cpp runtime
  - runtime metadata and benchmark harness explicitly report staged TurboQuant status

## Documentation Index

- `IMPLEMENTATION_PLAN.md`
- `docs/quickstart_linux.md`
- `docs/quickstart_wsl.md`
- `docs/hf_backend.md`
- `docs/gguf_backend.md`
- `docs/benchmarking.md`
- `docs/testing.md`
- `docs/limitations.md`

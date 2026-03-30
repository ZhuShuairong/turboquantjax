# GGUF Backend

## Overview

The GGUF backend supports real local inference for Qwen3.5 GGUF models using llama.cpp runtime paths:

- `llamacpp-python`
- `llamacpp-cli`
- `llama-server` (OpenAI-compatible local API)

## Commands

Generation:

```bash
tqjax-generate \
  --backend gguf \
  --model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --runtime llamacpp-python \
  --cache baseline \
  --prompt "Summarize the report in 5 bullets." \
  --max-new-tokens 64
```

Benchmark:

```bash
tqjax-bench \
  --backend gguf \
  --model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --runtime llamacpp-python \
  --contexts 2048,4096,8192,16384 \
  --cache-modes baseline,turboquant
```

Serve with llama-server:

```bash
tqjax-serve \
  --backend gguf \
  --model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080
```

## Staged TurboQuant Integration

Current milestone behavior:

- `baseline`: full production inference through llama.cpp runtime.
- `mse` and `turboquant`: staged mode with explicit metadata; generation still runs on robust llama.cpp execution path.

Why staged:

- full direct TurboQuant KV replacement inside upstream llama.cpp decode path is substantial engineering scope.
- this repo prioritizes immediate local usability while preserving a clear bridge for deeper integration.

## Runtime Requirements

- GGUF model file path must exist.
- One of the runtime options must be available:
  - `llama_cpp` Python package
  - `llama-cli` binary
  - `llama-server` binary for API mode

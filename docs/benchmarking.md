# Benchmarking

## Goal

Produce real end-to-end evidence for the tradeoff between baseline and compressed-cache modes.

## Primary Metrics

- model id/path
- backend (`hf` or `gguf`)
- runtime mode
- cache mode
- context length
- prompt length
- generation length
- key/value bit settings
- TTFT
- prefill tok/s
- decode tok/s
- wall time
- KV bytes and compression ratio (where available)
- peak RSS / GPU memory (best effort)
- quality score (retrieval/fidelity checks)

## HF Benchmark Example

```bash
tqjax-bench \
  --backend hf \
  --model Qwen/Qwen3.5-2B \
  --contexts 2048,4096,8192,16384,32768,65536,131072 \
  --cache-modes baseline,mse,turboquant \
  --export-json artifacts/hf_bench.json \
  --export-md artifacts/hf_bench.md
```

## GGUF Benchmark Example

```bash
tqjax-bench \
  --backend gguf \
  --model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --runtime llamacpp-python \
  --contexts 2048,4096,8192,16384,32768,65536,131072 \
  --cache-modes baseline,turboquant \
  --export-json artifacts/gguf_bench.json \
  --export-md artifacts/gguf_bench.md
```

## Matrix Runner

```bash
python scripts/benchmark_matrix.py \
  --hf-model Qwen/Qwen3.5-2B \
  --gguf-model-path /models/Qwen3.5-9B-Q4_K_M.gguf \
  --contexts 2048,4096,8192,16384
```

## Interpretation

- Expect baseline to remain faster in some regimes.
- Compression-only wins are not enough; prioritize full wall-clock and quality outcomes.
- For GGUF staged TurboQuant mode, treat reported metadata and comparison output as integration progress evidence, not full decode-path replacement.

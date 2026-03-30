# TurboQuant JAX Productionization Implementation Plan

## Scope

Transform the current research-heavy TurboQuant JAX repository into a production-ready local Qwen3.5 inference toolkit with:

- real Hugging Face generation path,
- real GGUF generation path (llama.cpp/llama-server bridge),
- long-context benchmark and quality harnesses,
- reproducible CLI and scripts for Linux + WSL2,
- robust unit + integration test coverage inspired by the PyTorch TurboQuant repos.

## Current Architecture Audit

### What exists

- Core quantization math and codebook generation in `turboquant_jax/turboquant.py`, `turboquant_jax/lloyd_max.py`, `turboquant_jax/quantization_utils.py`.
- Key/value compressors and tiled/fused score kernels in `turboquant_jax/compressors.py`, `turboquant_jax/fused_kernels.py`.
- Compatibility wrappers that mirror PyTorch class names in `turboquant_jax/compat.py`.
- Synthetic and partial real-model validation in `tests/test_turboquant.py` and `tests/validate.py`.
- Benchmark scripts that estimate/compare KV behavior but do not provide a unified production runtime UX.

### Confirmed defects/gaps

1. No cohesive runtime package for end-to-end generation with interchangeable backends.
2. Existing Qwen benchmark scripts rely on synthetic KV tensors for primary comparisons.
3. No single user-facing CLI for generation/bench/serve/validate.
4. No production GGUF runtime abstraction; existing scripts are benchmark-centric.
5. Existing tests focus heavily on algorithm math and limited validation scripts rather than complete inference regression matrix.
6. Documentation and install flow emphasize research benchmarking, not production local inference workflow.

## Hot Path and Bottleneck Summary

### Known bottlenecks

- Host-side orchestration and Python-level control flow in compression/benchmark loops.
- Repeated conversion between runtime tensor formats and framework boundaries.
- Missing streamlined runtime cache abstraction for prefill/decode loops.

### Immediate remediation strategy

- Keep compressed representation device-resident in JAX paths.
- Reuse prepared cache path when reuse is high; avoid repeated unpack.
- Provide typed runtime cache interfaces to remove ad hoc Python object reshaping.
- Add microbench hooks for: pack, unpack, quantize, score, prefill, decode.

## Target Runtime Architecture

## Package additions

- `turboquant_jax/runtime/__init__.py`
- `turboquant_jax/runtime/cache.py`
- `turboquant_jax/runtime/hf_backend.py`
- `turboquant_jax/runtime/gguf_backend.py`
- `turboquant_jax/runtime/llamacpp_bridge.py`
- `turboquant_jax/runtime/generate.py`
- `turboquant_jax/runtime/bench.py`
- `turboquant_jax/runtime/quality_eval.py`
- `turboquant_jax/runtime/telemetry.py`
- `turboquant_jax/runtime/serve.py`
- `turboquant_jax/runtime/cli.py`

## Runtime model

- Unified backend interface:
  - `generate(...)`
  - `benchmark(...)`
  - `quality_eval(...)`
  - `healthcheck(...)`
- HF backend:
  - baseline mode (native cache)
  - TurboQuant analysis mode (capture/quantize KV and compare score fidelity)
  - staged direct-cache integration mode where backend internals allow safe interception
- GGUF backend:
  - direct local inference via llama.cpp Python binding (when available)
  - fallback bridge to `llama-cli`/`llama-server` subprocess/API mode
  - staged TurboQuant integration path with explicit blockers and instrumentation

## CLI and User Workflow

- `tqjax-generate`
- `tqjax-bench`
- `tqjax-serve`
- `tqjax-validate`

All commands share common options:

- backend (`hf`, `gguf`)
- cache mode (`baseline`, `mse`, `turboquant`)
- bit-width controls (`bits-k`, `bits-v`)
- prompt source (`--prompt`, `--prompt-file`)
- export options (`--export-json`, `--export-md`)

## Benchmark Matrix

### Real generation benchmark dimensions

- Contexts: 2K, 4K, 8K, 16K, 32K, 64K, 128K (best effort 256K)
- Backends: HF baseline, HF TurboQuant, GGUF baseline, GGUF staged TurboQuant
- Metrics:
  - TTFT
  - prefill tok/s
  - decode tok/s
  - end-to-end wall time
  - KV bytes and compression ratio
  - peak memory/VRAM where measurable
  - quality score (retrieval/top-k overlap)

### Prompt sets

- hidden-fact retrieval
- long distractor summarization
- code lookup
- long-context Q&A
- lightweight reasoning
- structured extraction

## Test Plan

### Unit tests

- Lloyd-Max validity and symmetry
- quantization shape and determinism
- pack/unpack roundtrip and bit accounting
- cache serialization roundtrip

### Numerical validation

- MSE bounds and distortion tracking
- inner-product error/correlation
- top-k overlap and ranking preservation

### Runtime smoke/integration tests

- HF generate smoke (dependency-gated)
- GGUF generate smoke (dependency-gated)
- CLI smoke for generate/bench/validate
- long-context retrieval regression harness

### CI strategy

- CPU-only mandatory lane: unit + fast smoke
- optional GPU lane: HF runtime + extended bench smoke
- optional WSL scripts for local validation

## Platform Support Notes

### Linux

- Primary development/benchmark platform.
- CUDA and CPU paths supported.

### Windows 11 + WSL2

- Windows shell helpers call into WSL for runtime checks.
- WSL scripts verify CUDA visibility, JAX device visibility, and llama.cpp availability.

## Migration Plan

1. Add runtime package with stable interfaces and telemetry structs.
2. Add HF backend baseline generation and TurboQuant score-fidelity instrumentation.
3. Add GGUF backend through llama.cpp bridge and benchmark harness.
4. Add CLI and scripts that expose runtime functionality consistently.
5. Expand tests and add dependency-gated integration tests.
6. Overhaul docs and README with production workflows and limitations.
7. Final verification: installability, CLI smoke, benchmark artifact generation.

## Risks and staged handling

- Full direct TurboQuant injection into llama.cpp decode kernel is large-scope.
  - Stage 1: production GGUF baseline runtime + benchmark harness.
  - Stage 2: KV accounting and quality instrumentation.
  - Stage 3: TurboQuant-capable bridge/fork integration path.
- HF internals vary by transformers version; direct cache interception can be brittle.
  - Provide robust baseline path and clearly mark advanced injection mode as staged/experimental.

## Exit Criteria

The milestone is complete when:

- `pip install -e .` works.
- CLI commands run with clear diagnostics.
- HF backend performs real generation and produces benchmark artifacts.
- GGUF backend performs real local generation and produces benchmark artifacts.
- Unit tests and smoke tests pass in CPU environment.
- Docs include Linux and WSL quickstart with reproducible commands and known limitations.

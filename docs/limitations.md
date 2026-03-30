# Limitations

## 1) HF cache integration mode is analysis-first

In this milestone, HF `mse` and `turboquant` modes run real generation and compress real cache traces for fidelity and memory analysis.
Direct replacement of model-internal cache with compressed structures is not yet complete across all transformer variants.

## 2) GGUF TurboQuant decode replacement is staged

GGUF baseline inference is production-ready through llama.cpp runtime paths.
`mse` and `turboquant` GGUF modes are staged and report integration metadata while generation remains on the robust llama.cpp path.

## 3) Runtime variability across llama.cpp builds

Support for cache type flags and performance characteristics can differ by llama.cpp build, platform, and GPU driver.

## 4) Model/hardware dependency for long-context scale

Very large contexts (64K/128K/256K) depend on available VRAM/RAM and model architecture constraints.

## 5) Quality checks are benchmark harnesses, not formal eval suites

Built-in quality checks are lightweight retrieval/containment probes intended for regression signals, not final model quality certification.

## 6) Pallas kernel availability depends on backend support

Pallas scoring fallback is automatically handled, but availability is hardware/JAX-version dependent.

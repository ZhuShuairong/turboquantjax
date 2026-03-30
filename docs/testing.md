# Testing

## Default test lane

Run on CPU without external models:

```bash
pytest -q
```

Includes:

- codebook validation
- quantization math checks
- pack/unpack roundtrip
- cache roundtrip and compression accounting
- CLI parser and environment command smoke
- quality scoring utility tests

## Integration tests

Integration tests are opt-in and require local model/runtime setup.

### HF smoke

```bash
TQJAX_TEST_HF_MODEL=Qwen/Qwen3.5-2B \
pytest -q -m integration tests/test_hf_inference_smoke.py
```

### GGUF smoke

```bash
TQJAX_TEST_GGUF_MODEL=/models/Qwen3.5-9B-Q4_K_M.gguf \
pytest -q -m integration tests/test_gguf_inference_smoke.py
```

Optional variables:

- `TQJAX_TEST_HF_DEVICE`
- `TQJAX_TEST_GGUF_RUNTIME`
- `TQJAX_TEST_GGUF_SERVER_URL`

## WSL Validation

PowerShell launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_wsl_setup_check.ps1
```

## Suggested CI split

- CPU lane (required): all non-integration tests
- GPU lane (optional): selected HF integration tests
- GGUF lane (optional): GGUF smoke with local runtime image

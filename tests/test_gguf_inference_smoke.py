from __future__ import annotations

import os

import pytest

from turboquant_jax.runtime.generate import GenerateRequest
from turboquant_jax.runtime.gguf_backend import GgufBackend
from turboquant_jax.runtime.llamacpp_bridge import LlamaCppBridge


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("TQJAX_TEST_GGUF_MODEL"), reason="Set TQJAX_TEST_GGUF_MODEL to run GGUF smoke test")
def test_gguf_generation_smoke() -> None:
    model_path = os.environ["TQJAX_TEST_GGUF_MODEL"]
    runtime = os.environ.get("TQJAX_TEST_GGUF_RUNTIME", "llamacpp-python")

    backend = GgufBackend(bridge=LlamaCppBridge())
    request = GenerateRequest(
        backend="gguf",
        cache="baseline",
        model_path=model_path,
        runtime=runtime,
        prompt="Write a short sentence about local inference.",
        max_new_tokens=8,
        context_length=2048,
        extra={"server_url": os.environ.get("TQJAX_TEST_GGUF_SERVER_URL")},
    )

    result = backend.generate(request)
    assert result.output_text
    assert result.metrics.generated_tokens > 0

from __future__ import annotations

import os

import pytest

from turboquant_jax.runtime.generate import GenerateRequest
from turboquant_jax.runtime.hf_backend import HfBackend


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("TQJAX_TEST_HF_MODEL"), reason="Set TQJAX_TEST_HF_MODEL to run HF smoke test")
def test_hf_generation_smoke() -> None:
    model = os.environ["TQJAX_TEST_HF_MODEL"]

    backend = HfBackend(device=os.environ.get("TQJAX_TEST_HF_DEVICE", "auto"), dtype="auto")
    request = GenerateRequest(
        backend="hf",
        cache="baseline",
        model=model,
        prompt="Say hello in one sentence.",
        max_new_tokens=8,
    )

    result = backend.generate(request)

    assert result.output_text
    assert result.metrics.generated_tokens > 0
    assert result.metrics.wall_time_s > 0.0


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("TQJAX_TEST_HF_MODEL"), reason="Set TQJAX_TEST_HF_MODEL to run HF smoke test")
def test_hf_turboquant_cache_analysis_smoke() -> None:
    model = os.environ["TQJAX_TEST_HF_MODEL"]

    backend = HfBackend(device=os.environ.get("TQJAX_TEST_HF_DEVICE", "auto"), dtype="auto")
    request = GenerateRequest(
        backend="hf",
        cache="turboquant",
        model=model,
        prompt="Explain what caching means in language model inference.",
        max_new_tokens=6,
        bits_k=3,
        bits_v=2,
        extra={"max_eval_layers": 2},
    )

    result = backend.generate(request)
    assert result.metadata.get("cache_summary") is not None

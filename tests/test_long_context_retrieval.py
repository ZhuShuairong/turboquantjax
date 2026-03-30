from __future__ import annotations

from turboquant_jax.runtime.quality_eval import default_quality_cases, evaluate_quality_cases, evaluate_text_contains


def test_retrieval_scoring_simple_hit() -> None:
    out = evaluate_text_contains("The answer is AURORA-7749.", ["AURORA-7749"])
    assert out["score"] == 1.0
    assert out["hits"] == ["AURORA-7749"]


def test_quality_case_aggregate() -> None:
    cases = default_quality_cases()
    outputs = {case.name: "AURORA-7749 summary" for case in cases}
    report = evaluate_quality_cases(outputs, cases)
    assert 0.0 <= report["aggregate_score"] <= 1.0
    assert len(report["details"]) == len(cases)

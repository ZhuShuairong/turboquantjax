from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QualityCase:
    name: str
    prompt: str
    expected_substrings: list[str]


def default_quality_cases() -> list[QualityCase]:
    needle = "AURORA-7749"
    filler = (
        "Project status notes mention budget items, migration tasks, test timelines, and risk logs. "
        "This text is distractor context for retrieval validation. "
    )
    haystack = (filler * 200) + f" Hidden fact: the release codename is {needle}. " + (filler * 200)

    return [
        QualityCase(
            name="needle_retrieval",
            prompt=(
                "Read the document and answer with the exact codename.\n"
                f"Document:\n{haystack}\n"
                "Question: What is the release codename?"
            ),
            expected_substrings=[needle],
        ),
        QualityCase(
            name="long_summary",
            prompt=(
                "Summarize this long note in two bullet points:\n"
                + (filler * 250)
            ),
            expected_substrings=["-", "summary"],
        ),
    ]


def evaluate_text_contains(text: str, expected_substrings: list[str]) -> dict[str, Any]:
    text_norm = text.lower()
    hits = [substring for substring in expected_substrings if substring.lower() in text_norm]
    score = float(len(hits)) / float(len(expected_substrings) or 1)
    return {
        "score": score,
        "hits": hits,
        "misses": [s for s in expected_substrings if s not in hits],
    }


def evaluate_quality_cases(outputs: dict[str, str], cases: list[QualityCase]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for case in cases:
        text = outputs.get(case.name, "")
        result = evaluate_text_contains(text, case.expected_substrings)
        details.append(
            {
                "case": case.name,
                "score": result["score"],
                "hits": result["hits"],
                "misses": result["misses"],
            }
        )

    aggregate = float(sum(item["score"] for item in details) / max(len(details), 1))
    return {
        "aggregate_score": aggregate,
        "details": details,
    }

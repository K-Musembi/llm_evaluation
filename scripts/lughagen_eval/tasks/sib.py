from __future__ import annotations

from typing import Any, Iterable

from ..core_types import Example, ModelOutput, ScoreResult
from .base import Task

LANGUAGES = {
    "eng_Latn": "English",
    "swh_Latn": "Swahili",
    "luo_Latn": "Luo",
    "kik_Latn": "Kikuyu",
    "kam_Latn": "Kamba",
}

LABELS = [
    "science/technology", "travel", "politics", "sports", "health", "entertainment", "geography",
]


def _load_raw_rows(language_code: str, limit: int | None, cache_dir: str | None) -> Iterable[dict[str, Any]]:
    """Isolated so tests can monkeypatch this instead of hitting the Hub."""
    from datasets import load_dataset  # deferred: only needed for real runs

    dataset = load_dataset("Davlan/sib200", language_code, split="test", cache_dir=cache_dir)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield {"index": index, "text": row["text"], "category": row["category"]}


def normalize_response(response: str) -> str | None:
    normalized = response.casefold().strip()
    matches: list[tuple[int, str]] = []
    for label in LABELS:
        alternatives = [label]
        if label == "science/technology":
            alternatives += ["science and technology", "science technology"]
        for alternative in alternatives:
            position = normalized.find(alternative)
            if position >= 0:
                matches.append((position, label))
    return min(matches)[1] if matches else None


class SibTask(Task):
    name = "sib"
    scoring_mode = "generate_match"

    def load_examples(self, language: str, limit: int | None, cache_dir: str | None) -> Iterable[Example]:
        for row in _load_raw_rows(language, limit, cache_dir):
            yield Example(index=row["index"], source=row["text"], target=row["category"])

    def build_prompt(self, example: Example, language: str) -> str:
        # AfroBench's SIB prompt 3, with the language name filled in.
        language_name = LANGUAGES[language]
        return (
            "You are an assistant able to classify topics in texts.\n\n"
            "Given the categories science/technology, travel, politics, sports, health, "
            "entertainment, or geography; what is the topic of the " + language_name +
            " statement below? Return only the category.\n\n"
            "text: " + example.source + "\ncategory:"
        )

    def score(self, example: Example, output: ModelOutput) -> ScoreResult:
        raw = output.generated_text or ""
        normalized = normalize_response(raw)
        correct = normalized == example.target
        return ScoreResult(
            item_score=1.0 if correct else 0.0,
            correct=correct,
            normalized_prediction=normalized,
            score_type="exact category accuracy",
        )

    def aggregate(self, results: list[ScoreResult]) -> dict[str, Any]:
        if not results:
            return {"score_type": "accuracy", "score": None}
        accuracy = sum(1 for r in results if r.correct) / len(results)
        return {"score_type": "accuracy", "score": accuracy}

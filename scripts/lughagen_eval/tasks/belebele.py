"""Belebele: multiple-choice reading comprehension over FLORES-200 passages.

Dataset: ``facebook/belebele`` on the Hugging Face Hub, one config per FLORES-200
language code, single ``test`` split of 900 questions.

Of this project's four target languages, only Swahili (``swh_Latn``) and Luo
(``luo_Latn``) are among Belebele's 122 translated language variants; Kikuyu
and Kamba are not, and are reported as unsupported rather than silently
skipped (see ``supported_languages``).

Scored by log-likelihood over the four answer letters, following the
evaluation setting Belebele's own paper recommends for models without
task-specific finetuning: score the four candidate letters and take the
highest-probability one, rather than generating free text and string-matching
it. This avoids penalizing a model for not perfectly following an "only
output the letter" instruction when the underlying comprehension may be fine.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..local_scoring import best_choice_index, softmax
from ..core_types import Example, ModelOutput, ScoreResult
from .base import Task
from .sib import LANGUAGES

CHOICE_LETTERS = ["A", "B", "C", "D"]

# Of this pipeline's four target languages, Belebele only has these two.
SUPPORTED_LANGUAGES = {"swh_Latn", "luo_Latn"}


def _load_raw_rows(language_code: str, limit: int | None, cache_dir: str | None) -> Iterable[dict[str, Any]]:
    """Isolated so tests can monkeypatch this instead of hitting the Hub."""
    from datasets import load_dataset  # deferred: only needed for real runs

    dataset = load_dataset("facebook/belebele", language_code, split="test", cache_dir=cache_dir)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield {
            "index": index,
            "passage": row["flores_passage"],
            "question": row["question"],
            "choices": [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]],
            "correct_answer_num": str(row["correct_answer_num"]),
        }


class BelebeleTask(Task):
    name = "belebele"
    scoring_mode = "loglik_choice"

    def supported_languages(self) -> set[str] | None:
        return SUPPORTED_LANGUAGES

    def load_examples(self, language: str, limit: int | None, cache_dir: str | None) -> Iterable[Example]:
        if language not in SUPPORTED_LANGUAGES:
            return
        for row in _load_raw_rows(language, limit, cache_dir):
            correct_letter = CHOICE_LETTERS[int(row["correct_answer_num"]) - 1]
            yield Example(
                index=row["index"],
                source=row["question"],
                target=correct_letter,
                meta={"passage": row["passage"], "choices": row["choices"]},
            )

    def build_prompt(self, example: Example, language: str) -> str:
        passage = example.meta["passage"]
        choices = example.meta["choices"]
        lines = [f"P: {passage}", f"Q: {example.source}"]
        lines += [f"{letter}: {choice}" for letter, choice in zip(CHOICE_LETTERS, choices)]
        lines.append("Answer:")
        return "\n".join(lines)

    def choices(self, example: Example) -> list[str] | None:
        return CHOICE_LETTERS

    def score(self, example: Example, output: ModelOutput) -> ScoreResult:
        if output.choice_logprobs is None:
            return ScoreResult(item_score=0.0, correct=False, normalized_prediction=None,
                                score_type="4-way choice accuracy")
        best_index = best_choice_index(output.choice_logprobs)
        predicted_letter = CHOICE_LETTERS[best_index]
        correct = predicted_letter == example.target
        return ScoreResult(
            item_score=1.0 if correct else 0.0,
            correct=correct,
            normalized_prediction=predicted_letter,
            score_type="4-way choice accuracy",
        )

    def aggregate(self, results: list[ScoreResult]) -> dict[str, Any]:
        if not results:
            return {"score_type": "accuracy", "score": None}
        accuracy = sum(1 for r in results if r.correct) / len(results)
        return {"score_type": "accuracy", "score": accuracy, "random_baseline": 0.25}

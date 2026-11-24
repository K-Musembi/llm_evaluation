"""IrokoBench: AfriMMLU (knowledge QA), AfriXNLI (entailment), AfriMGSM (math).

Datasets (Hugging Face Hub): ``masakhane/afrimmlu``, ``masakhane/afrixnli``,
``masakhane/afrimgsm``, one config per language using IrokoBench's own short
language codes (e.g. ``swa`` for Swahili), *not* FLORES-200 codes.

Of this project's four target languages, IrokoBench only covers Swahili.
Luo, Kikuyu, and Kamba are not among its 16 languages, and unlike Belebele
this isn't a translation gap you can patch cheaply -- IrokoBench's value is
being *human*-translated, and MMLU/XNLI-style items are riskier to machine-
translate than a reading-comprehension question. Report Swahili-only for
this benchmark rather than running the other three languages through it.

Schema note: the exact ``masakhane/afrimgsm`` and ``masakhane/afrixnli``
column layouts used below reflect the public dataset viewer and the
lm-evaluation-harness IrokoBench task configs at the time this was written.
Before a real run, confirm columns with e.g.::

    python -c "from datasets import load_dataset; \
        print(load_dataset('masakhane/afrimgsm', 'swa', split='test')[0])"

and adjust ``_load_raw_rows`` below if the live schema has since changed.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from ..local_scoring import best_choice_index
from ..core_types import Example, ModelOutput, ScoreResult
from .base import Task

# This project's FLORES-200 codes -> IrokoBench's own language codes.
# Only Swahili is covered; the mapping is intentionally this small.
FLORES_TO_IROKO = {"swh_Latn": "swa"}
SUPPORTED_LANGUAGES = set(FLORES_TO_IROKO)

MMLU_CHOICE_LETTERS = ["A", "B", "C", "D"]
XNLI_LABELS = ["entailment", "neutral", "contradiction"]  # standard XNLI label order (0, 1, 2)


# --------------------------------------------------------------------------
# AfriMMLU -- multiple-choice knowledge QA
# --------------------------------------------------------------------------

def _load_afrimmlu_rows(language_code: str, limit: int | None, cache_dir: str | None) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("masakhane/afrimmlu", language_code, split="test", cache_dir=cache_dir)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        choices = row["choices"]
        if isinstance(choices, str):  # some exports store this as a stringified list
            import ast

            choices = ast.literal_eval(choices)
        yield {"index": index, "question": row["question"], "choices": choices, "answer": row["answer"]}


class AfriMmluTask(Task):
    name = "afrimmlu"
    scoring_mode = "loglik_choice"

    def supported_languages(self) -> set[str] | None:
        return SUPPORTED_LANGUAGES

    def load_examples(self, language: str, limit: int | None, cache_dir: str | None) -> Iterable[Example]:
        if language not in SUPPORTED_LANGUAGES:
            return
        iroko_code = FLORES_TO_IROKO[language]
        for row in _load_afrimmlu_rows(iroko_code, limit, cache_dir):
            yield Example(
                index=row["index"], source=row["question"], target=row["answer"],
                meta={"choices": row["choices"]},
            )

    def build_prompt(self, example: Example, language: str) -> str:
        choices = example.meta["choices"]
        lines = [f"Question: {example.source}"]
        lines += [f"{letter}: {choice}" for letter, choice in zip(MMLU_CHOICE_LETTERS, choices)]
        lines.append("Answer:")
        return "\n".join(lines)

    def choices(self, example: Example) -> list[str] | None:
        return MMLU_CHOICE_LETTERS

    def score(self, example: Example, output: ModelOutput) -> ScoreResult:
        if output.choice_logprobs is None:
            return ScoreResult(item_score=0.0, correct=False, normalized_prediction=None,
                                score_type="4-way choice accuracy")
        predicted_letter = MMLU_CHOICE_LETTERS[best_choice_index(output.choice_logprobs)]
        correct = predicted_letter == example.target
        return ScoreResult(item_score=1.0 if correct else 0.0, correct=correct,
                            normalized_prediction=predicted_letter, score_type="4-way choice accuracy")

    def aggregate(self, results: list[ScoreResult]) -> dict[str, Any]:
        if not results:
            return {"score_type": "accuracy", "score": None}
        accuracy = sum(1 for r in results if r.correct) / len(results)
        return {"score_type": "accuracy", "score": accuracy, "random_baseline": 0.25}


# --------------------------------------------------------------------------
# AfriXNLI -- natural language inference
# --------------------------------------------------------------------------

def _load_afrixnli_rows(language_code: str, limit: int | None, cache_dir: str | None) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("masakhane/afrixnli", language_code, split="test", cache_dir=cache_dir)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        label = row["label"]
        label_index = int(label) if not isinstance(label, str) else XNLI_LABELS.index(label.lower())
        yield {"index": index, "premise": row["premise"], "hypothesis": row["hypothesis"], "label_index": label_index}


class AfriXnliTask(Task):
    name = "afrixnli"
    scoring_mode = "loglik_choice"

    def supported_languages(self) -> set[str] | None:
        return SUPPORTED_LANGUAGES

    def load_examples(self, language: str, limit: int | None, cache_dir: str | None) -> Iterable[Example]:
        if language not in SUPPORTED_LANGUAGES:
            return
        iroko_code = FLORES_TO_IROKO[language]
        for row in _load_afrixnli_rows(iroko_code, limit, cache_dir):
            yield Example(
                index=row["index"], source=row["premise"], target=XNLI_LABELS[row["label_index"]],
                meta={"hypothesis": row["hypothesis"]},
            )

    def build_prompt(self, example: Example, language: str) -> str:
        return (
            f"Premise: {example.source}\n"
            f"Hypothesis: {example.meta['hypothesis']}\n"
            "Question: Does the premise entail the hypothesis, contradict it, or neither?\n"
            "Answer:"
        )

    def choices(self, example: Example) -> list[str] | None:
        return XNLI_LABELS

    def score(self, example: Example, output: ModelOutput) -> ScoreResult:
        if output.choice_logprobs is None:
            return ScoreResult(item_score=0.0, correct=False, normalized_prediction=None,
                                score_type="3-way NLI accuracy")
        predicted = XNLI_LABELS[best_choice_index(output.choice_logprobs)]
        correct = predicted == example.target
        return ScoreResult(item_score=1.0 if correct else 0.0, correct=correct,
                            normalized_prediction=predicted, score_type="3-way NLI accuracy")

    def aggregate(self, results: list[ScoreResult]) -> dict[str, Any]:
        if not results:
            return {"score_type": "accuracy", "score": None}
        accuracy = sum(1 for r in results if r.correct) / len(results)
        return {"score_type": "accuracy", "score": accuracy, "random_baseline": 1 / 3}


# --------------------------------------------------------------------------
# AfriMGSM -- grade-school math word problems (generation, not choice-scored)
# --------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text: str) -> float | None:
    """Pull the last number out of a free-text answer, tolerating commas."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    cleaned = matches[-1].replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _load_afrimgsm_rows(language_code: str, limit: int | None, cache_dir: str | None) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("masakhane/afrimgsm", language_code, split="test", cache_dir=cache_dir)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield {"index": index, "question": row["question"], "answer_number": row["answer_number"]}


class AfriMgsmTask(Task):
    name = "afrimgsm"
    scoring_mode = "generate_extract"

    def supported_languages(self) -> set[str] | None:
        return SUPPORTED_LANGUAGES

    def load_examples(self, language: str, limit: int | None, cache_dir: str | None) -> Iterable[Example]:
        if language not in SUPPORTED_LANGUAGES:
            return
        iroko_code = FLORES_TO_IROKO[language]
        for row in _load_afrimgsm_rows(iroko_code, limit, cache_dir):
            yield Example(index=row["index"], source=row["question"], target=str(row["answer_number"]))

    def build_prompt(self, example: Example, language: str) -> str:
        return f"Question: {example.source}\nSolve step by step, then give the final numeric answer.\nAnswer:"

    def score(self, example: Example, output: ModelOutput) -> ScoreResult:
        predicted_number = extract_final_number(output.generated_text or "")
        try:
            target_number = float(example.target)
        except ValueError:
            target_number = None
        correct = (
            predicted_number is not None and target_number is not None
            and abs(predicted_number - target_number) < 1e-6
        )
        return ScoreResult(
            item_score=1.0 if correct else 0.0, correct=correct,
            normalized_prediction=None if predicted_number is None else str(predicted_number),
            score_type="exact numeric-answer accuracy",
        )

    def aggregate(self, results: list[ScoreResult]) -> dict[str, Any]:
        if not results:
            return {"score_type": "accuracy", "score": None}
        accuracy = sum(1 for r in results if r.correct) / len(results)
        return {"score_type": "accuracy", "score": accuracy}

from __future__ import annotations

from typing import Any, Iterable

from ..core_types import Example, ModelOutput, ScoreResult
from .base import Task
from .sib import LANGUAGES  # shared FLORES-200 language-name table


def _load_raw_rows(
    source_code: str, target_code: str, limit: int | None, cache_dir: str | None, dataset_name: str
) -> Iterable[dict[str, Any]]:
    """Isolated so tests can monkeypatch this instead of hitting the Hub."""
    from datasets import load_dataset  # deferred: only needed for real runs

    pair_config = f"{source_code}-{target_code}"
    dataset = load_dataset(dataset_name, pair_config, split="devtest", cache_dir=cache_dir)
    source_column = f"sentence_{source_code}"
    target_column = f"sentence_{target_code}"
    missing = {source_column, target_column} - set(dataset.column_names)
    if missing:
        raise ValueError(f"FLORES config {pair_config!r} is missing expected columns: {sorted(missing)}")
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield {"index": index, "source": row[source_column], "target": row[target_column]}


class FloresTask(Task):
    name = "flores"
    scoring_mode = "generate_match"

    def __init__(self, source_code: str, target_code: str, dataset_name: str = "facebook/flores"):
        self.source_code = source_code
        self.target_code = target_code
        self.dataset_name = dataset_name

    @property
    def direction_key(self) -> str:
        return f"{self.source_code}-to-{self.target_code}"

    def load_examples(self, language: str, limit: int | None, cache_dir: str | None) -> Iterable[Example]:
        # ``language`` is accepted for interface uniformity but the source
        # and target codes are fixed at construction (one task instance per
        # translation direction).
        for row in _load_raw_rows(self.source_code, self.target_code, limit, cache_dir, self.dataset_name):
            yield Example(index=row["index"], source=row["source"], target=row["target"])

    def build_prompt(self, example: Example, language: str) -> str:
        # AfroBench's FLORES prompt 1, with the language names filled in.
        source_name = LANGUAGES[self.source_code]
        target_name = LANGUAGES[self.target_code]
        return (
            f"{source_name} sentence: {example.source}\n"
            f"{target_name} sentence: . Return only the translated sentence."
        )

    def score(self, example: Example, output: ModelOutput) -> ScoreResult:
        import sacrebleu  # deferred: keeps this module importable without the dependency

        prediction = (output.generated_text or "").strip()
        item_score = sacrebleu.sentence_chrf(prediction, [example.target]).score
        return ScoreResult(item_score=item_score, correct=None, normalized_prediction=None, score_type="sentence ChrF")

    def aggregate(self, results: list[ScoreResult]) -> dict[str, Any]:
        # The headline FLORES metric is corpus-level chrF, not the mean of
        # per-sentence scores; the runner passes predictions/targets alongside
        # results via ``aggregate_corpus`` below for that reason.
        if not results:
            return {"score_type": "corpus ChrF", "score": None}
        mean_sentence_chrf = sum(r.item_score for r in results) / len(results)
        return {"score_type": "mean sentence ChrF (see corpus ChrF for the headline score)", "score": mean_sentence_chrf}

    def aggregate_corpus(self, predictions: list[str], references: list[str]) -> float:
        import sacrebleu

        return sacrebleu.corpus_chrf(predictions, [references]).score

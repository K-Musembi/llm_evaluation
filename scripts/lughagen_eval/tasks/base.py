"""The ``Task`` interface every benchmark plugs into.

A task owns three responsibilities: turning a raw dataset row into an
``Example``, turning an ``Example`` into a prompt, and turning a model's
``ModelOutput`` into a ``ScoreResult``. The runner (``lughagen_eval.runner``)
is written only against this interface, so adding a new benchmark never
requires touching the runner, the JSONL logging, or the resumability logic.

``scoring_mode`` tells the runner which backend method to call:

* ``"generate_match"``  -- generate free text, compare/score it directly
  (SIB: exact-match label; FLORES: chrF against the reference).
* ``"generate_extract"`` -- generate free text, but only after pulling a
  specific piece out of it (AfriMGSM: the final numeric answer).
* ``"loglik_choice"``   -- score fixed candidate continuations by
  log-likelihood and take the best one (Belebele, AfriMMLU, AfriXNLI).
  This is the fairer mode for a model that has not been heavily
  instruction-tuned: see the discussion in scripts/README.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from ..core_types import Example, ModelOutput, ScoreResult

SCORING_MODES = ("generate_match", "generate_extract", "loglik_choice")


class Task(ABC):
    name: str
    scoring_mode: str

    @abstractmethod
    def load_examples(self, language: str, limit: int | None, cache_dir: str | None) -> Iterable[Example]:
        """Yield ``Example`` rows for one language, in a stable order."""

    @abstractmethod
    def build_prompt(self, example: Example, language: str) -> str:
        """Return the exact text sent to the model for this example."""

    def choices(self, example: Example) -> list[str] | None:
        """Candidate continuations for ``loglik_choice`` tasks; else ``None``."""
        return None

    @abstractmethod
    def score(self, example: Example, output: ModelOutput) -> ScoreResult:
        """Turn one model output into a per-item score."""

    @abstractmethod
    def aggregate(self, results: list[ScoreResult]) -> dict[str, Any]:
        """Combine per-item scores into the job-level headline metric(s)."""

    def supported_languages(self) -> set[str] | None:
        """FLORES-200 language codes this task has data for, or ``None`` if unrestricted.

        Used by the runner to skip (and clearly report, rather than silently
        drop) a language/task combination the underlying dataset doesn't
        cover -- e.g. Belebele has no Kikuyu or Kamba, and IrokoBench only
        has Swahili among this project's four target languages.
        """
        return None

"""Shared data types for the generalized LughaGen evaluation framework.

Kept in their own module (rather than inside ``tasks`` or ``backends``) so
that both can import them without creating a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding / request parameters shared by every backend."""

    max_new_tokens: int = 256
    temperature: float = 0.01
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    timeout: int = 90
    retries: int = 3
    api_language: str = "Auto"


@dataclass
class Example:
    """One evaluation item, in a shape generic enough for every task.

    ``source`` is whatever text is shown to the model (a sentence, or a
    passage + question already merged, depending on the task). ``target`` is
    the reference the prediction is scored against (a label, a translation,
    or a correct-choice letter). ``meta`` carries anything task-specific a
    prompt builder or scorer needs (e.g. the four MCQ choices).
    """

    index: int
    source: str
    target: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """What a backend returns for one request."""

    generated_text: str | None = None
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    choice_logprobs: list[float] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ScoreResult:
    """What a task's ``score`` method returns for one example."""

    item_score: float
    correct: bool | None
    normalized_prediction: str | None
    score_type: str

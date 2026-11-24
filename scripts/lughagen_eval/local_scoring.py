"""Pure-Python helpers for log-likelihood multiple-choice scoring.

These functions intentionally take and return plain Python floats/lists
rather than tensors, so they can be unit-tested without ``torch`` or
``transformers`` installed. ``backends.LocalBackend`` calls these after
converting model output tensors to plain lists with ``.tolist()``.
"""

from __future__ import annotations

import math


def sum_target_logprobs(token_logprobs: list[float]) -> float:
    """Sum per-token log-probabilities for one candidate continuation.

    ``token_logprobs`` is the log-probability the model assigned to each
    actual continuation token, in order (i.e. already gathered from the
    model's per-position log-softmax at the correct vocabulary index).
    Summing log-probabilities is equivalent to multiplying probabilities,
    which is the standard way to score a multi-token continuation.
    """
    if not token_logprobs:
        raise ValueError("token_logprobs must contain at least one value")
    return float(sum(token_logprobs))


def average_target_logprob(token_logprobs: list[float]) -> float:
    """Length-normalized variant of :func:`sum_target_logprobs`.

    Useful when candidate continuations have different token lengths (e.g.
    comparing "A" vs "entertainment"): raw sums penalize longer strings, so
    some evaluation setups prefer the per-token average instead.
    """
    if not token_logprobs:
        raise ValueError("token_logprobs must contain at least one value")
    return float(sum(token_logprobs) / len(token_logprobs))


def best_choice_index(choice_scores: list[float]) -> int:
    """Return the index of the highest-scoring choice (ties -> first)."""
    if not choice_scores:
        raise ValueError("choice_scores must not be empty")
    best_index = 0
    best_value = choice_scores[0]
    for i, value in enumerate(choice_scores[1:], start=1):
        if value > best_value:
            best_value = value
            best_index = i
    return best_index


def split_context_continuation(context_ids: list[int], full_ids: list[int]) -> tuple[list[int], list[int]]:
    """Split ``full_ids`` (tokenization of prompt + delimiter + choice) into the
    context the model conditions on and the continuation tokens to score.

    Tokenizing the prompt and the choice separately is a classic source of
    silent errors: e.g. a Llama-3 style tokenizer turns "A" into a different
    token than " A" (the one the model actually expects after "Answer:").
    So the caller tokenizes the *whole* string and this function finds where
    it diverges from the prompt-only tokenization. Normally that is exactly at
    ``len(context_ids)``; if the last prompt token merged with the start of the
    choice, the context is shortened so the merged token is scored too.
    """
    limit = min(len(context_ids), len(full_ids))
    shared = 0
    while shared < limit and context_ids[shared] == full_ids[shared]:
        shared += 1
    if shared == 0:
        raise ValueError("prompt and prompt+choice tokenizations share no prefix")
    continuation = list(full_ids[shared:])
    if not continuation:
        raise ValueError("choice produced no continuation tokens")
    return list(full_ids[:shared]), continuation


def score_pairs(pairs: list[tuple[list[int], list[int]]], row_fn) -> list[float]:
    """Log-likelihood of each ``(context_ids, continuation_ids)`` pair.

    ``row_fn(ids)`` must return ``rows`` where ``rows[i][t]`` is
    ``log P(token t | ids[: i + 1])`` -- a nested list in tests, a
    ``log_softmax`` tensor on the GPU in ``LocalBackend``. Keeping the model
    call behind this callback lets the token bookkeeping be unit-tested
    without torch.

    Fast path: when every choice is a single token after an identical context
    (the A/B/C/D letters of Belebele and AfriMMLU), ONE forward pass over the
    prompt scores all of them, instead of one pass per choice.
    """
    if not pairs:
        raise ValueError("pairs must not be empty")
    first_context = pairs[0][0]
    if all(context == first_context and len(cont) == 1 for context, cont in pairs):
        last_row = row_fn(first_context)[len(first_context) - 1]
        return [float(last_row[cont[0]]) for _, cont in pairs]
    scores: list[float] = []
    for context, cont in pairs:
        rows = row_fn(context + cont)
        scores.append(sum_target_logprobs([float(rows[len(context) - 1 + j][tok]) for j, tok in enumerate(cont)]))
    return scores


def softmax(values: list[float]) -> list[float]:
    """Numerically-stable softmax, used to report choice probabilities."""
    if not values:
        raise ValueError("values must not be empty")
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]

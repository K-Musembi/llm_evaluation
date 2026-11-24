import math

import pytest

from lughagen_eval.local_scoring import (
    average_target_logprob,
    best_choice_index,
    softmax,
    sum_target_logprobs,
)


def test_sum_target_logprobs_basic():
    assert sum_target_logprobs([-0.1, -0.2, -0.3]) == pytest.approx(-0.6)


def test_sum_target_logprobs_rejects_empty():
    with pytest.raises(ValueError):
        sum_target_logprobs([])


def test_average_target_logprob_normalizes_by_length():
    # A long, individually-confident continuation shouldn't lose to a short,
    # less-confident one just because raw sums penalize length -- that's the
    # whole point of the length-normalized variant.
    long_continuation = [-0.1] * 10   # sum = -1.0, avg = -0.1
    short_continuation = [-0.5]        # sum = -0.5, avg = -0.5
    assert sum_target_logprobs(short_continuation) > sum_target_logprobs(long_continuation)
    assert average_target_logprob(long_continuation) > average_target_logprob(short_continuation)


def test_best_choice_index_picks_max():
    assert best_choice_index([-5.0, -1.0, -3.0, -9.0]) == 1


def test_best_choice_index_ties_pick_first():
    assert best_choice_index([-1.0, -1.0, -2.0]) == 0


def test_best_choice_index_rejects_empty():
    with pytest.raises(ValueError):
        best_choice_index([])


def test_softmax_sums_to_one_and_is_monotonic():
    probs = softmax([1.0, 2.0, 3.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[2] > probs[1] > probs[0]


def test_softmax_is_numerically_stable_for_large_values():
    # Naive exp() without the max-subtraction trick would overflow here.
    probs = softmax([1000.0, 1001.0, 999.0])
    assert sum(probs) == pytest.approx(1.0)
    assert all(math.isfinite(p) for p in probs)

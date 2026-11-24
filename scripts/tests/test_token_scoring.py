import math

import pytest

from lughagen_eval.local_scoring import score_pairs, split_context_continuation


# ---- split_context_continuation -------------------------------------------

def test_split_normal_case_continuation_follows_context():
    context, cont = split_context_continuation([1, 2, 3], [1, 2, 3, 9])
    assert context == [1, 2, 3] and cont == [9]


def test_split_multi_token_continuation():
    context, cont = split_context_continuation([1, 2, 3], [1, 2, 3, 7, 8])
    assert context == [1, 2, 3] and cont == [7, 8]


def test_split_when_last_context_token_merges_with_choice():
    # The tokenizer fused the prompt's last token with the start of the choice:
    # the shared prefix is shorter than the prompt, so the merged token is scored.
    context, cont = split_context_continuation([1, 2, 3], [1, 2, 7, 8])
    assert context == [1, 2] and cont == [7, 8]


def test_split_rejects_identical_tokenization():
    with pytest.raises(ValueError):
        split_context_continuation([1, 2, 3], [1, 2, 3])


def test_split_rejects_no_shared_prefix():
    with pytest.raises(ValueError):
        split_context_continuation([1, 2, 3], [9, 8, 7])


# ---- score_pairs with a toy language model ----------------------------------

VOCAB = 6


def make_row_fn(table):
    """rows[i] = log-probs over the vocab given only ids[i] (a bigram 'model')."""
    calls = []

    def row_fn(ids):
        calls.append(list(ids))
        rows = []
        for token in ids:
            probs = table[token]
            rows.append([math.log(p) for p in probs])
        return rows

    row_fn.calls = calls
    return row_fn


def uniform():
    return [1 / VOCAB] * VOCAB


def test_single_token_choices_use_one_forward_pass():
    # After context ending in token 3, the model puts 70% on token 4 and 10% on each of 0,1,2,5.
    table = {i: uniform() for i in range(VOCAB)}
    table[3] = [0.1, 0.1, 0.1, 0.0 + 1e-9, 0.7, 0.1 - 1e-9]
    row_fn = make_row_fn(table)
    pairs = [([1, 3], [0]), ([1, 3], [4]), ([1, 3], [5]), ([1, 3], [2])]
    scores = score_pairs(pairs, row_fn)
    assert len(row_fn.calls) == 1  # all four choices scored from ONE pass over the prompt
    assert scores.index(max(scores)) == 1  # the choice token 4 wins
    assert scores[1] == pytest.approx(math.log(0.7))


def test_multi_token_choices_sum_log_probs_and_need_one_pass_each():
    table = {i: uniform() for i in range(VOCAB)}
    table[3] = [0.0 + 1e-9, 0.0 + 1e-9, 0.5, 0.0 + 1e-9, 0.5 - 3e-9, 0.0 + 1e-9]  # next: 2 or 4
    table[2] = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]                                  # after 2: mostly 0
    table[4] = [0.02, 0.02, 0.02, 0.02, 0.02, 0.9]                                  # after 4: mostly 5
    row_fn = make_row_fn(table)
    pairs = [([1, 3], [2, 0]), ([1, 3], [4, 5]), ([1, 3], [2, 1])]
    scores = score_pairs(pairs, row_fn)
    assert len(row_fn.calls) == 3
    assert scores[0] == pytest.approx(math.log(0.5) + math.log(0.9))
    assert scores[1] == pytest.approx(math.log(0.5 - 3e-9) + math.log(0.9))
    assert scores[0] > scores[2]  # same first token, but "0" is far likelier than "1" after it


def test_score_pairs_differing_contexts_fall_back_to_per_choice_passes():
    table = {i: uniform() for i in range(VOCAB)}
    row_fn = make_row_fn(table)
    score_pairs([([1, 3], [4]), ([1, 2], [4])], row_fn)
    assert len(row_fn.calls) == 2


def test_score_pairs_rejects_empty_input():
    with pytest.raises(ValueError):
        score_pairs([], lambda ids: [])

from lughagen_eval.tasks import belebele as belebele_module
from lughagen_eval.tasks.belebele import CHOICE_LETTERS, BelebeleTask
from lughagen_eval.core_types import Example, ModelOutput


def test_belebele_supported_languages_excludes_kikuyu_and_kamba():
    task = BelebeleTask()
    supported = task.supported_languages()
    assert supported == {"swh_Latn", "luo_Latn"}
    assert "kik_Latn" not in supported
    assert "kam_Latn" not in supported


def test_belebele_load_examples_skips_unsupported_language():
    task = BelebeleTask()
    examples = list(task.load_examples("kik_Latn", None, None))
    assert examples == []


def test_belebele_load_examples_uses_injected_rows(monkeypatch):
    def fake_rows(language_code, limit, cache_dir):
        yield {
            "index": 0, "passage": "A short passage.", "question": "What happened?",
            "choices": ["Nothing", "Something", "Everything", "Anything"], "correct_answer_num": "2",
        }

    monkeypatch.setattr(belebele_module, "_load_raw_rows", fake_rows)
    task = BelebeleTask()
    examples = list(task.load_examples("swh_Latn", None, None))
    assert len(examples) == 1
    example = examples[0]
    assert example.target == "B"  # correct_answer_num 2 -> letter B
    assert example.meta["choices"][1] == "Something"


def test_belebele_build_prompt_lists_all_four_choices():
    task = BelebeleTask()
    example = Example(index=0, source="What happened?", target="B",
                       meta={"passage": "A short passage.", "choices": ["W", "X", "Y", "Z"]})
    prompt = task.build_prompt(example, "swh_Latn")
    assert "P: A short passage." in prompt
    assert "Q: What happened?" in prompt
    for letter, choice in zip(CHOICE_LETTERS, ["W", "X", "Y", "Z"]):
        assert f"{letter}: {choice}" in prompt
    assert prompt.strip().endswith("Answer:")


def test_belebele_choices_returns_four_letters():
    task = BelebeleTask()
    example = Example(index=0, source="q", target="A", meta={"passage": "p", "choices": ["a", "b", "c", "d"]})
    assert task.choices(example) == ["A", "B", "C", "D"]


def test_belebele_score_picks_highest_logprob_choice():
    task = BelebeleTask()
    example = Example(index=0, source="q", target="C", meta={"passage": "p", "choices": ["a", "b", "c", "d"]})
    # Choice C (index 2) has the highest (least negative) log-likelihood.
    output = ModelOutput(choice_logprobs=[-5.0, -4.0, -0.5, -3.0])
    result = task.score(example, output)
    assert result.normalized_prediction == "C"
    assert result.correct is True
    assert result.item_score == 1.0


def test_belebele_score_wrong_choice():
    task = BelebeleTask()
    example = Example(index=0, source="q", target="A", meta={"passage": "p", "choices": ["a", "b", "c", "d"]})
    output = ModelOutput(choice_logprobs=[-5.0, -0.5, -4.0, -3.0])  # B wins
    result = task.score(example, output)
    assert result.correct is False


def test_belebele_aggregate_reports_random_baseline():
    task = BelebeleTask()
    from lughagen_eval.core_types import ScoreResult

    results = [ScoreResult(item_score=1.0, correct=True, normalized_prediction="A", score_type="x")] * 3
    results += [ScoreResult(item_score=0.0, correct=False, normalized_prediction="B", score_type="x")]
    agg = task.aggregate(results)
    assert agg["score"] == 0.75
    assert agg["random_baseline"] == 0.25

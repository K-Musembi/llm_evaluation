from lughagen_eval.tasks import irokobench as iroko_module
from lughagen_eval.tasks.irokobench import (
    AfriMgsmTask,
    AfriMmluTask,
    AfriXnliTask,
    XNLI_LABELS,
    extract_final_number,
)
from lughagen_eval.core_types import Example, ModelOutput


def test_iroko_tasks_only_support_swahili():
    for task in (AfriMmluTask(), AfriXnliTask(), AfriMgsmTask()):
        assert task.supported_languages() == {"swh_Latn"}
        assert list(task.load_examples("luo_Latn", None, None)) == []
        assert list(task.load_examples("kik_Latn", None, None)) == []
        assert list(task.load_examples("kam_Latn", None, None)) == []


# --- AfriMMLU ---------------------------------------------------------------

def test_afrimmlu_load_examples_uses_injected_rows(monkeypatch):
    def fake_rows(language_code, limit, cache_dir):
        assert language_code == "swa"  # FLORES code translated to IrokoBench's own code
        yield {"index": 0, "question": "2+2=?", "choices": ["3", "4", "5", "6"], "answer": "B"}

    monkeypatch.setattr(iroko_module, "_load_afrimmlu_rows", fake_rows)
    task = AfriMmluTask()
    examples = list(task.load_examples("swh_Latn", None, None))
    assert examples[0].target == "B"
    assert examples[0].meta["choices"] == ["3", "4", "5", "6"]


def test_afrimmlu_build_prompt_and_score():
    task = AfriMmluTask()
    example = Example(index=0, source="2+2=?", target="B", meta={"choices": ["3", "4", "5", "6"]})
    prompt = task.build_prompt(example, "swh_Latn")
    assert "A: 3" in prompt and "B: 4" in prompt and prompt.strip().endswith("Answer:")
    result = task.score(example, ModelOutput(choice_logprobs=[-3.0, -0.1, -2.0, -4.0]))
    assert result.correct is True and result.normalized_prediction == "B"


# --- AfriXNLI ----------------------------------------------------------------

def test_afrixnli_load_examples_handles_integer_label(monkeypatch):
    # _load_afrixnli_rows (patched here) already does the int/string ->
    # label_index normalization internally; that's what load_examples expects.
    def fake_rows(language_code, limit, cache_dir):
        yield {"index": 0, "premise": "It is raining.", "hypothesis": "The weather is wet.", "label_index": 0}

    monkeypatch.setattr(iroko_module, "_load_afrixnli_rows", fake_rows)
    task = AfriXnliTask()
    examples = list(task.load_examples("swh_Latn", None, None))
    assert examples[0].target == "entailment"


def test_afrixnli_build_prompt_and_score():
    task = AfriXnliTask()
    example = Example(index=0, source="It is raining.", target="contradiction",
                       meta={"hypothesis": "It is sunny."})
    prompt = task.build_prompt(example, "swh_Latn")
    assert "Premise: It is raining." in prompt
    assert "Hypothesis: It is sunny." in prompt
    assert task.choices(example) == list(XNLI_LABELS)
    # contradiction is index 2 -> give it the highest logprob
    result = task.score(example, ModelOutput(choice_logprobs=[-3.0, -2.0, -0.1]))
    assert result.correct is True
    assert result.normalized_prediction == "contradiction"


def test_afrixnli_aggregate_random_baseline_is_one_third():
    task = AfriXnliTask()
    assert task.aggregate([])["score"] is None
    from lughagen_eval.core_types import ScoreResult

    results = [ScoreResult(1.0, True, "entailment", "x")]
    agg = task.aggregate(results)
    assert abs(agg["random_baseline"] - 1 / 3) < 1e-9


# --- AfriMGSM ----------------------------------------------------------------

def test_extract_final_number_plain():
    assert extract_final_number("The answer is 42.") == 42.0


def test_extract_final_number_with_commas_and_decimals():
    assert extract_final_number("Total cost: 1,234.5 shillings") == 1234.5


def test_extract_final_number_picks_last_when_multiple():
    assert extract_final_number("There were 3 apples and 5 oranges, total 8") == 8.0


def test_extract_final_number_none_when_no_digits():
    assert extract_final_number("I cannot compute this") is None


def test_afrimgsm_score_correct_and_incorrect():
    task = AfriMgsmTask()
    example = Example(index=0, source="2+2?", target="4")
    correct = task.score(example, ModelOutput(generated_text="Working it out... the answer is 4"))
    wrong = task.score(example, ModelOutput(generated_text="the answer is 5"))
    unparsable = task.score(example, ModelOutput(generated_text="no idea"))
    assert correct.correct is True
    assert wrong.correct is False
    assert unparsable.correct is False


def test_afrimgsm_load_examples_uses_injected_rows(monkeypatch):
    def fake_rows(language_code, limit, cache_dir):
        yield {"index": 0, "question": "2+2?", "answer_number": 4}

    monkeypatch.setattr(iroko_module, "_load_afrimgsm_rows", fake_rows)
    task = AfriMgsmTask()
    examples = list(task.load_examples("swh_Latn", None, None))
    assert examples[0].target == "4"

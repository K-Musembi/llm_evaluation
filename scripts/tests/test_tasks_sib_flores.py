from lughagen_eval.tasks import sib as sib_module
from lughagen_eval.tasks import flores as flores_module
from lughagen_eval.tasks.sib import SibTask, normalize_response
from lughagen_eval.tasks.flores import FloresTask
from lughagen_eval.core_types import Example, ModelOutput


def test_sib_normalize_response_exact_label():
    assert normalize_response("travel") == "travel"


def test_sib_normalize_response_picks_earliest_mentioned_label():
    # If a model rambles and mentions two labels, AfroBench's own scoring
    # takes whichever appears first in the text.
    assert normalize_response("this could be sports, but also politics") == "sports"


def test_sib_normalize_response_handles_science_technology_variants():
    assert normalize_response("Science and Technology") == "science/technology"
    assert normalize_response("science technology") == "science/technology"


def test_sib_normalize_response_no_match_returns_none():
    assert normalize_response("I have no idea") is None


def test_sib_build_prompt_contains_language_and_text():
    task = SibTask()
    example = Example(index=0, source="Bomu hii...", target="science/technology")
    prompt = task.build_prompt(example, "swh_Latn")
    assert "Swahili" in prompt
    assert "Bomu hii..." in prompt
    assert prompt.strip().endswith("category:")


def test_sib_score_correct_and_incorrect():
    task = SibTask()
    example = Example(index=0, source="x", target="travel")
    correct = task.score(example, ModelOutput(generated_text="travel"))
    wrong = task.score(example, ModelOutput(generated_text="sports"))
    assert correct.correct is True and correct.item_score == 1.0
    assert wrong.correct is False and wrong.item_score == 0.0


def test_sib_aggregate_accuracy():
    task = SibTask()
    results = [task.score(Example(0, "x", "travel"), ModelOutput(generated_text=t))
               for t in ["travel", "sports", "travel", "geography"]]
    agg = task.aggregate(results)
    assert agg["score_type"] == "accuracy"
    assert agg["score"] == 0.5  # 2 of 4 correct


def test_sib_load_examples_uses_injected_rows(monkeypatch):
    def fake_rows(language_code, limit, cache_dir):
        yield {"index": 0, "text": "hello", "category": "travel"}
        yield {"index": 1, "text": "world", "category": "sports"}

    monkeypatch.setattr(sib_module, "_load_raw_rows", fake_rows)
    task = SibTask()
    examples = list(task.load_examples("swh_Latn", None, None))
    assert len(examples) == 2
    assert examples[0].source == "hello" and examples[0].target == "travel"


def test_flores_build_prompt_and_direction_key():
    task = FloresTask("swh_Latn", "eng_Latn")
    assert task.direction_key == "swh_Latn-to-eng_Latn"
    example = Example(index=0, source="Habari yako", target="How are you")
    prompt = task.build_prompt(example, "swh_Latn")
    assert "Swahili sentence: Habari yako" in prompt
    assert "English sentence:" in prompt


def test_flores_score_uses_chrf():
    task = FloresTask("eng_Latn", "swh_Latn")
    example = Example(index=0, source="hello", target="Habari yako")
    good = task.score(example, ModelOutput(generated_text="Habari yako"))
    bad = task.score(example, ModelOutput(generated_text="kitu tofauti kabisa"))
    assert good.item_score > bad.item_score
    assert good.item_score > 90  # near-exact match should score very high chrF


def test_flores_aggregate_corpus_chrf_matches_sacrebleu():
    import sacrebleu

    task = FloresTask("eng_Latn", "swh_Latn")
    preds = ["Habari yako", "Jina langu ni Juma"]
    refs = ["Habari yako", "Jina langu ni Juma"]
    expected = sacrebleu.corpus_chrf(preds, [refs]).score
    assert task.aggregate_corpus(preds, refs) == expected


def test_flores_load_examples_uses_injected_rows(monkeypatch):
    def fake_rows(source_code, target_code, limit, cache_dir, dataset_name):
        yield {"index": 0, "source": "hi", "target": "habari"}

    monkeypatch.setattr(flores_module, "_load_raw_rows", fake_rows)
    task = FloresTask("eng_Latn", "swh_Latn")
    examples = list(task.load_examples("swh_Latn", None, None))
    assert examples[0].source == "hi" and examples[0].target == "habari"

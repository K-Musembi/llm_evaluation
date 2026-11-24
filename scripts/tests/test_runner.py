import json

import pytest

from lughagen_eval.backends import FakeBackend
from lughagen_eval.io_utils import read_ok_records
from lughagen_eval.runner import evaluate_job, summarize
from lughagen_eval.tasks import belebele as belebele_module
from lughagen_eval.tasks import sib as sib_module
from lughagen_eval.tasks.belebele import BelebeleTask
from lughagen_eval.tasks.sib import SibTask
from lughagen_eval.core_types import GenerationConfig


def _patch_sib_rows(monkeypatch, rows):
    def fake_rows(language_code, limit, cache_dir):
        for r in rows:
            yield r

    monkeypatch.setattr(sib_module, "_load_raw_rows", fake_rows)


def test_evaluate_job_generate_match_writes_expected_records(tmp_path, monkeypatch):
    _patch_sib_rows(monkeypatch, [
        {"index": 0, "text": "a", "category": "travel"},
        {"index": 1, "text": "b", "category": "sports"},
    ])
    backend = FakeBackend(generate_responses={"a\ncategory:": "travel", "b\ncategory:": "sports"})
    task = SibTask()
    path = evaluate_job(task=task, model="m", language="swh_Latn", limit=None,
                         output_dir=tmp_path, backend=backend, config=GenerationConfig())
    records = read_ok_records(path)
    assert len(records) == 2
    assert all(r["correct"] for r in records)

    # human-review + metrics files should exist alongside the JSONL
    assert path.with_suffix(".review.csv").exists()
    metrics = json.loads(path.with_suffix(".metrics.json").read_text())
    assert metrics["score"] == 1.0
    assert metrics["examples"] == 2


def test_evaluate_job_is_resumable_after_partial_failure(tmp_path, monkeypatch):
    _patch_sib_rows(monkeypatch, [
        {"index": 0, "text": "a", "category": "travel"},
        {"index": 1, "text": "b", "category": "sports"},
        {"index": 2, "text": "c", "category": "health"},
    ])
    # First run: index 1's prompt triggers a simulated failure.
    backend = FakeBackend(
        generate_responses={"a\ncategory:": "travel", "c\ncategory:": "health"},
        fail_on={"b\ncategory:"},
    )
    task = SibTask()
    path = evaluate_job(task=task, model="m", language="swh_Latn", limit=None,
                         output_dir=tmp_path, backend=backend, config=GenerationConfig())
    records = read_ok_records(path)
    assert {r["index"] for r in records} == {0, 2}  # index 1 failed, not recorded as ok

    # Second run ("resume"): same output dir, backend now succeeds on index 1.
    # Only the previously-failed index should be re-sent to the backend.
    backend2 = FakeBackend(generate_responses={"b\ncategory:": "sports"})
    evaluate_job(task=task, model="m", language="swh_Latn", limit=None,
                 output_dir=tmp_path, backend=backend2, config=GenerationConfig())
    resumed_records = read_ok_records(path)
    assert {r["index"] for r in resumed_records} == {0, 1, 2}
    # Only the missing index was actually re-queried against the backend.
    assert len(backend2.calls) == 1
    assert backend2.calls[0][0] == "generate"
    assert "text: b" in backend2.calls[0][1]


def test_evaluate_job_skips_unsupported_language_without_writing_a_file(tmp_path):
    task = BelebeleTask()
    backend = FakeBackend()
    path = evaluate_job(task=task, model="m", language="kik_Latn", limit=None,
                         output_dir=tmp_path, backend=backend, config=GenerationConfig())
    assert path is None
    assert not (tmp_path / "belebele").exists()


def test_evaluate_job_loglik_choice_dispatches_to_loglikelihood(tmp_path, monkeypatch):
    def fake_rows(language_code, limit, cache_dir):
        yield {
            "index": 0, "passage": "p", "question": "q?",
            "choices": ["w", "x", "y", "z"], "correct_answer_num": "3",
        }

    monkeypatch.setattr(belebele_module, "_load_raw_rows", fake_rows)
    task = BelebeleTask()
    backend = FakeBackend(loglik_responses={"Answer:": [-9.0, -8.0, -0.1, -7.0]})  # C wins
    path = evaluate_job(task=task, model="m", language="swh_Latn", limit=None,
                         output_dir=tmp_path, backend=backend, config=GenerationConfig())
    records = read_ok_records(path)
    assert records[0]["normalized_prediction"] == "C"
    assert records[0]["correct"] is True
    assert backend.calls == [("loglikelihood", backend.calls[0][1])]


def test_summarize_writes_one_row_per_job(tmp_path, monkeypatch):
    _patch_sib_rows(monkeypatch, [{"index": 0, "text": "a", "category": "travel"}])
    backend = FakeBackend(generate_responses={"category:": "travel"})
    task = SibTask()
    path = evaluate_job(task=task, model="model-a", language="swh_Latn", limit=None,
                         output_dir=tmp_path, backend=backend, config=GenerationConfig())
    summary_path = summarize([path], {"sib": task}, tmp_path)
    content = summary_path.read_text()
    assert "sib" in content and "model-a" in content and "swh_Latn" in content


def test_summarize_skips_none_paths_from_unsupported_languages(tmp_path):
    summary_path = summarize([None], {}, tmp_path)
    lines = summary_path.read_text().strip().splitlines()
    assert len(lines) == 1  # header only, no crash on the skipped job

"""Multi-session behaviour: time budget, resume, circuit breaker, ordering, status."""
import json

import pytest

from lughagen_eval.backends import FakeBackend
from lughagen_eval.budget import RunBudget
from lughagen_eval.io_utils import read_ok_records
from lughagen_eval.report import build_report
from lughagen_eval.runner import (
    AbortRun, evaluate_job, expected_job_keys, job_status, run_jobs, summarize,
)
from lughagen_eval.tasks import belebele as belebele_module
from lughagen_eval.tasks import sib as sib_module
from lughagen_eval.tasks.belebele import BelebeleTask
from lughagen_eval.tasks.sib import SibTask
from lughagen_eval.core_types import GenerationConfig, ModelOutput


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TickingBackend(FakeBackend):
    """Each request 'takes' one second on the fake clock."""

    def __init__(self, clock, **kwargs):
        super().__init__(**kwargs)
        self.clock = clock

    def generate(self, model, prompt, config):
        self.clock.t += 1
        return super().generate(model, prompt, config)


def patch_sib(monkeypatch, n):
    def rows(language_code, limit, cache_dir):
        for i in range(n):
            yield {"index": i, "text": f"t{i}", "category": "travel"}

    monkeypatch.setattr(sib_module, "_load_raw_rows", rows)


def test_budget_stops_mid_job_and_next_session_finishes_without_repeating_work(tmp_path, monkeypatch):
    patch_sib(monkeypatch, 5)
    task = SibTask()
    jobs = {"sib": [(task, "swh_Latn")]}

    # Session 1: the budget allows only 2 one-second requests.
    clock = FakeClock()
    backend1 = TickingBackend(clock, default_generate="travel")
    outcome = run_jobs(jobs, ["M"], backend=backend1, config=GenerationConfig(), output_dir=tmp_path,
                       budget=RunBudget(max_seconds=2, clock=clock))
    assert outcome.stopped == "budget"
    path = tmp_path / "sib" / "M" / "swh_Latn.jsonl"
    assert [r["index"] for r in read_ok_records(path)] == [0, 1]
    metrics = json.loads(path.with_suffix(".metrics.json").read_text())
    assert metrics["complete"] is False and metrics["examples"] == 2

    # Session 2 (a new process, no budget): only the 3 missing examples are requested.
    backend2 = FakeBackend(default_generate="travel")
    outcome2 = run_jobs(jobs, ["M"], backend=backend2, config=GenerationConfig(), output_dir=tmp_path)
    assert outcome2.stopped is None
    assert len(backend2.calls) == 3
    assert [r["index"] for r in read_ok_records(path)] == [0, 1, 2, 3, 4]
    assert json.loads(path.with_suffix(".metrics.json").read_text())["complete"] is True

    # Session 3: everything is done, so the backend (and hence the model load) is never touched.
    backend3 = FakeBackend()
    run_jobs(jobs, ["M"], backend=backend3, config=GenerationConfig(), output_dir=tmp_path)
    assert backend3.calls == []


def test_circuit_breaker_aborts_instead_of_logging_errors_for_the_whole_dataset(tmp_path, monkeypatch):
    patch_sib(monkeypatch, 100)
    backend = FakeBackend(fail_on={"text:"})  # every prompt fails (e.g. CUDA out of memory)
    with pytest.raises(AbortRun, match="consecutive backend errors"):
        evaluate_job(task=SibTask(), model="M", language="swh_Latn", output_dir=tmp_path, backend=backend,
                     config=GenerationConfig(), max_consecutive_errors=3)
    assert len(backend.calls) == 3  # stopped after 3, not 100
    lines = (tmp_path / "sib" / "M" / "swh_Latn.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3 and all(json.loads(line)["status"] == "error" for line in lines)


def test_run_jobs_reports_aborted_and_keeps_going_state_clean(tmp_path, monkeypatch):
    patch_sib(monkeypatch, 100)
    outcome = run_jobs({"sib": [(SibTask(), "swh_Latn")]}, ["M"], backend=FakeBackend(fail_on={"text:"}),
                       config=GenerationConfig(), output_dir=tmp_path)
    assert outcome.stopped == "aborted" and "consecutive" in outcome.message


def test_isolated_errors_do_not_abort_and_mark_the_job_incomplete(tmp_path, monkeypatch):
    patch_sib(monkeypatch, 5)
    backend = FakeBackend(default_generate="travel", fail_on={"text: t2\n"})
    path = evaluate_job(task=SibTask(), model="M", language="swh_Latn", output_dir=tmp_path, backend=backend,
                        config=GenerationConfig())
    assert [r["index"] for r in read_ok_records(path)] == [0, 1, 3, 4]
    assert json.loads(path.with_suffix(".metrics.json").read_text())["complete"] is False  # index 2 still missing


def test_models_run_one_at_a_time_so_each_is_loaded_once(tmp_path, monkeypatch):
    patch_sib(monkeypatch, 2)

    def belebele_rows(language_code, limit, cache_dir):
        yield {"index": 0, "passage": "p", "question": "q", "choices": list("wxyz"), "correct_answer_num": "1"}

    monkeypatch.setattr(belebele_module, "_load_raw_rows", belebele_rows)
    jobs = {"sib": [(SibTask(), "swh_Latn"), (SibTask(), "luo_Latn")],
            "belebele": [(BelebeleTask(), "swh_Latn")]}
    backend = FakeBackend(default_generate="travel")
    run_jobs(jobs, ["ModelA", "ModelB"], backend=backend, config=GenerationConfig(), output_dir=tmp_path)
    order = backend.model_calls
    first_b = order.index("ModelB")
    assert set(order[:first_b]) == {"ModelA"} and set(order[first_b:]) == {"ModelB"}  # no A-B-A switching
    assert order.count("ModelA") == order.count("ModelB") == 5  # (2 + 2 sib) + 1 belebele


def test_expected_job_keys_skip_languages_the_dataset_does_not_cover():
    jobs = {"belebele": [(BelebeleTask(), lang) for lang in ["swh_Latn", "luo_Latn", "kik_Latn", "kam_Latn"]]}
    keys = expected_job_keys(jobs, ["M"])
    assert keys == [("belebele", "M", "swh_Latn"), ("belebele", "M", "luo_Latn")]


def test_job_status_reports_complete_partial_and_not_started(tmp_path, monkeypatch):
    patch_sib(monkeypatch, 4)
    jobs = {"sib": [(SibTask(), "swh_Latn"), (SibTask(), "luo_Latn"), (SibTask(), "kik_Latn")]}
    # finish swh, leave luo partial via a tiny budget, never touch kik
    evaluate_job(task=SibTask(), model="M", language="swh_Latn", output_dir=tmp_path,
                 backend=FakeBackend(default_generate="travel"), config=GenerationConfig())
    clock = FakeClock()
    evaluate_job(task=SibTask(), model="M", language="luo_Latn", output_dir=tmp_path,
                 backend=TickingBackend(clock, default_generate="travel"), config=GenerationConfig(),
                 budget=RunBudget(max_seconds=1, clock=clock))
    status = {row["language"]: row["status"] for row in job_status(tmp_path, expected_job_keys(jobs, ["M"]))}
    assert status == {"swh_Latn": "complete", "luo_Latn": "partial", "kik_Latn": "not_started"}


def test_partial_jobs_are_flagged_in_summary_and_dashboard(tmp_path, monkeypatch):
    patch_sib(monkeypatch, 4)
    clock = FakeClock()
    task = SibTask()
    path = evaluate_job(task=task, model="M", language="swh_Latn", output_dir=tmp_path,
                        backend=TickingBackend(clock, default_generate="travel"), config=GenerationConfig(),
                        budget=RunBudget(max_seconds=2, clock=clock))
    summary = summarize([path], {"sib": task}, tmp_path)
    assert "complete" in summary.read_text().splitlines()[0]
    assert ",False," in summary.read_text()
    html = build_report([summary], tmp_path / "dashboard.html").read_text()
    assert "partial" in html


def test_budget_syncs_during_and_after_a_job(tmp_path, monkeypatch):
    patch_sib(monkeypatch, 3)
    synced = []
    clock = FakeClock()
    budget = RunBudget(sync_fn=lambda: synced.append(clock.t), sync_interval_seconds=1, clock=clock)
    evaluate_job(task=SibTask(), model="M", language="swh_Latn", output_dir=tmp_path,
                 backend=TickingBackend(clock, default_generate="travel"), config=GenerationConfig(), budget=budget)
    assert len(synced) >= 2  # at least one periodic sync plus the forced one at the end of the job

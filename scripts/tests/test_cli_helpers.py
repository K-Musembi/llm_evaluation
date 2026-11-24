import argparse
import json

import pytest

import run_eval
from lughagen_eval.backends import APIBackend, BackendError
from lughagen_eval.tasks.flores import FloresTask
from lughagen_eval.tasks.sib import SibTask


def test_build_task_instances_single_instance_task_shares_one_task_across_languages():
    jobs = run_eval.build_task_instances(["sib"], ["swh_Latn", "luo_Latn"], "both", "facebook/flores")
    tasks_used = {id(t) for t, _ in jobs["sib"]}
    assert len(tasks_used) == 1  # one shared SibTask instance
    assert isinstance(jobs["sib"][0][0], SibTask)
    assert [lang for _, lang in jobs["sib"]] == ["swh_Latn", "luo_Latn"]


def test_build_task_instances_flores_both_directions():
    jobs = run_eval.build_task_instances(["flores"], ["swh_Latn"], "both", "facebook/flores")
    directions = [t.direction_key for t, _ in jobs["flores"]]
    assert set(directions) == {"swh_Latn-to-eng_Latn", "eng_Latn-to-swh_Latn"}


def test_build_task_instances_flores_to_english_only():
    jobs = run_eval.build_task_instances(["flores"], ["swh_Latn", "luo_Latn"], "to-english", "facebook/flores")
    directions = [t.direction_key for t, _ in jobs["flores"]]
    assert directions == ["swh_Latn-to-eng_Latn", "luo_Latn-to-eng_Latn"]


def test_build_task_instances_rejects_unknown_task():
    with pytest.raises(SystemExit):
        run_eval.build_task_instances(["not-a-real-task"], ["swh_Latn"], "both", "facebook/flores")


def test_build_backend_api_ignores_model_paths_file():
    args = argparse.Namespace(backend="api", endpoint="https://x/generate", score_endpoint=None,
                               model_paths_file=None, device="cuda")
    backend = run_eval.build_backend(args)
    assert isinstance(backend, APIBackend)
    assert backend.endpoint == "https://x/generate"


def test_build_backend_local_requires_model_paths_file():
    args = argparse.Namespace(backend="local", endpoint="x", score_endpoint=None,
                               model_paths_file=None, device="cuda")
    with pytest.raises(SystemExit):
        run_eval.build_backend(args)


def test_build_backend_local_reads_model_paths_json(tmp_path, monkeypatch):
    paths_file = tmp_path / "models.json"
    paths_file.write_text(json.dumps({"Model-A": "/models/model-a"}))
    args = argparse.Namespace(backend="local", endpoint="x", score_endpoint=None,
                               model_paths_file=str(paths_file), device="cpu")
    # LocalBackend itself requires torch/transformers; assert we get that far
    # (i.e. JSON parsing + dispatch works) and fail only on the missing deps.
    with pytest.raises(BackendError):
        run_eval.build_backend(args)

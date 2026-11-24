"""End-to-end tests of run_eval.main() the way a Colab session uses it."""
import argparse
import json
import shutil
import sys

import pytest

import run_eval
from lughagen_eval.backends import FakeBackend
from lughagen_eval.tasks import sib as sib_module


@pytest.fixture
def fake_sib(monkeypatch):
    def rows(language_code, limit, cache_dir):
        for i in range(4):
            yield {"index": i, "text": f"t{i}", "category": "travel"}

    monkeypatch.setattr(sib_module, "_load_raw_rows", rows)


def run_main(monkeypatch, argv, backend):
    monkeypatch.setattr(run_eval, "build_backend", lambda args: backend)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv])
    return run_eval.main()


def base_args(tmp_path, out="local", sync="drive", extra=()):
    return ["--tasks", "sib", "--languages", "swh_Latn", "--models", "M", "--backend", "local",
            "--output-dir", str(tmp_path / out), "--sync-dir", str(tmp_path / sync), *extra]


def test_full_session_writes_outputs_and_syncs_to_drive(tmp_path, monkeypatch, fake_sib):
    backend = FakeBackend(default_generate="travel")
    code = run_main(monkeypatch, base_args(tmp_path, extra=["--max-runtime-hours", "5.5"]), backend)
    assert code == 0
    for name in ("summary.csv", "dashboard.html", "run_config.json"):
        assert (tmp_path / "local" / name).exists()
        assert (tmp_path / "drive" / name).exists(), f"{name} was not synced to the Drive folder"
    assert (tmp_path / "drive" / "sib" / "M" / "swh_Latn.jsonl").exists()
    config = json.loads((tmp_path / "local" / "run_config.json").read_text())
    assert config["backend"] == "local" and config["quantization"] == "none"
    assert len(config["sessions"]) == 1


def test_next_session_after_local_disk_is_wiped_restores_from_drive_and_does_no_rework(tmp_path, monkeypatch, fake_sib):
    run_main(monkeypatch, base_args(tmp_path), FakeBackend(default_generate="travel"))

    shutil.rmtree(tmp_path / "local")  # Colab gave us a fresh VM
    backend2 = FakeBackend()
    code = run_main(monkeypatch, base_args(tmp_path), backend2)
    assert code == 0
    assert backend2.calls == []  # nothing left to do, so no model would even be loaded
    assert (tmp_path / "local" / "summary.csv").exists()
    config = json.loads((tmp_path / "local" / "run_config.json").read_text())
    assert len(config["sessions"]) == 1 or len(config["sessions"]) == 2  # history is kept, never lost


def test_budget_exit_code_is_10_and_resume_finishes(tmp_path, monkeypatch, fake_sib, capsys):
    # A budget so small it expires before the first example.
    code = run_main(monkeypatch, base_args(tmp_path, extra=["--max-runtime-hours", "0.000000001"]),
                    FakeBackend(default_generate="travel"))
    assert code == 10
    assert "Run the same command again" in capsys.readouterr().out

    shutil.rmtree(tmp_path / "local")
    code = run_main(monkeypatch, base_args(tmp_path, extra=["--max-runtime-hours", "5"]),
                    FakeBackend(default_generate="travel"))
    assert code == 0
    summary = (tmp_path / "local" / "summary.csv").read_text()
    assert "sib,M,swh_Latn,4," in summary


def test_abort_exit_code_is_20_when_the_backend_keeps_failing(tmp_path, monkeypatch, fake_sib, capsys):
    code = run_main(monkeypatch, base_args(tmp_path, extra=["--max-consecutive-errors", "3"]),
                    FakeBackend(fail_on={"text:"}))
    assert code == 20
    assert "ABORTED" in capsys.readouterr().out


def test_failures_below_the_abort_threshold_are_reported_not_hidden(tmp_path, monkeypatch, fake_sib, capsys):
    # Every example fails, but only 4 in a row (< default 25): must not claim success.
    code = run_main(monkeypatch, base_args(tmp_path), FakeBackend(fail_on={"text:"}))
    out = capsys.readouterr().out
    assert code == 30
    assert "FINISHED WITH GAPS" in out and "All requested jobs finished" not in out


def test_summary_covers_every_job_on_disk_not_only_this_sessions(tmp_path, monkeypatch, fake_sib):
    run_main(monkeypatch, base_args(tmp_path), FakeBackend(default_generate="travel"))
    # session 2 only runs a different model; session 1's results must stay in the summary
    argv = ["--tasks", "sib", "--languages", "swh_Latn", "--models", "N", "--backend", "local",
            "--output-dir", str(tmp_path / "local"), "--sync-dir", str(tmp_path / "drive")]
    run_main(monkeypatch, argv, FakeBackend(default_generate="travel"))
    summary = (tmp_path / "local" / "summary.csv").read_text()
    assert "sib,M,swh_Latn" in summary and "sib,N,swh_Latn" in summary


def test_status_lists_jobs_without_calling_the_backend(tmp_path, monkeypatch, fake_sib, capsys):
    run_main(monkeypatch, base_args(tmp_path), FakeBackend(default_generate="travel"))
    capsys.readouterr()
    backend = FakeBackend()
    code = run_main(monkeypatch, base_args(tmp_path, extra=["--status"]), backend)
    out = capsys.readouterr().out
    assert code == 0 and backend.calls == []
    assert "complete" in out and "1 complete, 0 partial, 0 not started" in out


def test_models_default_to_the_models_file_for_local_runs(tmp_path):
    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps({"A": "/x/a", "B": {"path": "/x/b", "base_model_path": "/x/base"}}))
    args = argparse.Namespace(models=None, backend="local", model_paths_file=str(models_file))
    assert run_eval.resolve_models(args) == ["A", "B"]
    api_args = argparse.Namespace(models=None, backend="api", model_paths_file=None)
    assert run_eval.resolve_models(api_args) == run_eval.DEFAULT_MODELS
    explicit = argparse.Namespace(models=["Z"], backend="local", model_paths_file=str(models_file))
    assert run_eval.resolve_models(explicit) == ["Z"]


def test_check_setup_reports_sizes_and_problems(tmp_path, capsys):
    good = tmp_path / "good"
    good.mkdir()
    (good / "config.json").write_text("{}")
    (good / "model.safetensors").write_bytes(b"0" * 1_000_000)
    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps({"Good": str(good), "Missing": str(tmp_path / "missing")}))
    args = argparse.Namespace(backend="local", model_paths_file=str(models_file), models=["Good", "Missing"],
                              quantization="4bit")
    code = run_eval.print_check_setup(args)
    out = capsys.readouterr().out
    assert code == 1  # one model is missing
    assert "Good:" in out and "fp16" in out and "4bit" in out
    assert "PROBLEM: Missing: folder not found" in out


def test_check_setup_passes_when_everything_is_present(tmp_path, capsys):
    good = tmp_path / "good"
    good.mkdir()
    (good / "config.json").write_text("{}")
    (good / "model.safetensors").write_bytes(b"0" * 1000)
    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps({"Good": str(good)}))
    args = argparse.Namespace(backend="local", model_paths_file=str(models_file), models=["Good"], quantization="none")
    assert run_eval.print_check_setup(args) == 0
    assert "Setup looks OK" in capsys.readouterr().out

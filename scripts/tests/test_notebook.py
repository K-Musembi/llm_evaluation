"""The Colab notebook can't be run here (no Colab, no Drive), so verify what can be:
every code cell is valid Python, no secret is embedded, and the arguments the
cells build are accepted by the real command-line parsers."""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import run_eval

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "colab_eval.ipynb"


@pytest.fixture(scope="module")
def cells():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in data["cells"]]


def code_cells(cells):
    return [source for source, kind in cells if kind == "code"]


def test_notebook_is_valid_nbformat():
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))


def test_every_code_cell_is_valid_python(cells):
    for source in code_cells(cells):
        ast.parse(source)  # raises SyntaxError if a cell is broken


def test_no_secrets_or_outputs_are_embedded(cells):
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|hf_[A-Za-z0-9]{20,}", text)
    data = json.loads(text)
    assert all(not c.get("outputs") for c in data["cells"] if c["cell_type"] == "code")


def test_notebook_reads_tokens_from_colab_secrets_not_from_text(cells):
    joined = "\n".join(code_cells(cells))
    assert 'userdata.get(secret)' in joined and '"GITHUB_TOKEN"' in joined and '"HF_TOKEN"' in joined


def namespace_with_settings_and_helpers(cells):
    sources = code_cells(cells)
    settings = next(s for s in sources if "MAX_RUNTIME_HOURS" in s and "OUTPUT_DIR" in s)
    helpers = next(s for s in sources if "def eval_args" in s)
    namespace = {"os": os, "sys": sys, "subprocess": subprocess}
    exec(settings, namespace)
    exec(helpers, namespace)
    return namespace


def test_arguments_built_by_the_notebook_are_accepted_by_run_eval(cells):
    ns = namespace_with_settings_and_helpers(cells)
    parser = run_eval.build_parser()
    for extra in ([], ["--check-setup"], ["--status"], ["--max-runtime-hours", str(ns["MAX_RUNTIME_HOURS"]), "--limit", "5"]):
        args = parser.parse_args([str(a) for a in ns["eval_args"]() + extra])
        assert args.backend == "local" and args.quantization == ns["QUANTIZATION"]
        assert str(args.sync_dir) == ns["SYNC_DIR"] and str(args.output_dir) == ns["OUTPUT_DIR"]
        assert args.tasks == ns["TASKS"]


def test_notebook_models_override_is_passed_through(cells):
    ns = namespace_with_settings_and_helpers(cells)
    ns["MODELS"] = ["Llama-8B-FFT"]
    args = run_eval.build_parser().parse_args([str(a) for a in ns["eval_args"]()])
    assert args.models == ["Llama-8B-FFT"]


def test_every_flag_mentioned_in_the_notebook_exists(cells):
    known = set(run_eval.build_parser()._option_string_actions)
    publish_flags = {"--results-dir", "--repo-dir", "--run-name", "--dry-run"}   # publish_results.py
    git_flags = {"--branch", "--rebase"}                                          # the clone/pull cell
    mentioned = set(re.findall(r'"(--[a-z][a-z0-9-]+)"', "\n".join(code_cells(cells))))
    assert mentioned, "expected the notebook to pass some flags"
    unknown = mentioned - known - publish_flags - git_flags
    assert not unknown, unknown


def test_notebook_publish_arguments_match_the_publish_cli(cells):
    ns = namespace_with_settings_and_helpers(cells)
    result = subprocess.run(
        [sys.executable, str(Path(run_eval.__file__).parent / "publish_results.py"),
         "--results-dir", "/nonexistent", "--repo-dir", ns["REPO_DIR"], "--run-name", ns["RUN_NAME"], "--dry-run"],
        capture_output=True, text=True)
    assert "unrecognized arguments" not in result.stderr
    assert "required" not in result.stderr


def test_notebook_default_task_order_runs_the_weak_tasks_last(cells):
    ns = namespace_with_settings_and_helpers(cells)
    assert ns["TASKS"][-2:] == ["flores", "sib"]
    assert ns["TASKS"][0] == "belebele"
    assert 0 < ns["MAX_RUNTIME_HOURS"] <= 6

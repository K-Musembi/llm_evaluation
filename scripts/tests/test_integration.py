from pathlib import Path

from lughagen_eval.backends import FakeBackend
from lughagen_eval.report import build_report
from lughagen_eval.runner import evaluate_job, summarize
from lughagen_eval.tasks import belebele as belebele_module
from lughagen_eval.tasks import sib as sib_module
from lughagen_eval.tasks.belebele import BelebeleTask
from lughagen_eval.tasks.sib import SibTask
from lughagen_eval.core_types import GenerationConfig

ALL_LANGUAGES = ["swh_Latn", "luo_Latn", "kik_Latn", "kam_Latn"]
MODELS = ["Model-A", "Model-B"]


def test_full_run_across_tasks_models_languages(tmp_path, monkeypatch):
    def fake_sib_rows(language_code, limit, cache_dir):
        yield {"index": 0, "text": "hello", "category": "travel"}

    def fake_belebele_rows(language_code, limit, cache_dir):
        yield {"index": 0, "passage": "p", "question": "q?",
               "choices": ["w", "x", "y", "z"], "correct_answer_num": "1"}

    monkeypatch.setattr(sib_module, "_load_raw_rows", fake_sib_rows)
    monkeypatch.setattr(belebele_module, "_load_raw_rows", fake_belebele_rows)

    sib_task = SibTask()
    belebele_task = BelebeleTask()
    generate_backend = FakeBackend(generate_responses={"category:": "travel"})
    loglik_backend = FakeBackend(loglik_responses={"Answer:": [-0.1, -5.0, -5.0, -5.0]})  # A wins

    paths = []
    for model in MODELS:
        for language in ALL_LANGUAGES:
            paths.append(evaluate_job(task=sib_task, model=model, language=language,
                                       output_dir=tmp_path, backend=generate_backend,
                                       config=GenerationConfig()))
            paths.append(evaluate_job(task=belebele_task, model=model, language=language,
                                       output_dir=tmp_path, backend=loglik_backend,
                                       config=GenerationConfig()))

    # Belebele should have skipped Kikuyu and Kamba for both models -> None paths.
    none_count = sum(1 for p in paths if p is None)
    assert none_count == 4  # 2 models x (Kikuyu, Kamba)

    summary_path = summarize(paths, {"sib": sib_task, "belebele": belebele_task}, tmp_path)
    rows = summary_path.read_text().strip().splitlines()
    # header + (2 models x 4 langs sib) + (2 models x 2 langs belebele) = 1 + 8 + 4
    assert len(rows) == 13

    dashboard_path = build_report([summary_path], tmp_path / "dashboard.html")
    html_content = dashboard_path.read_text()
    assert "Model-A" in html_content and "Model-B" in html_content
    assert 'title="not covered by this dataset"' in html_content  # belebele kik/kam gap is visible

import csv
from pathlib import Path

from lughagen_eval.report import build_report, build_coverage_matrix, render_html


SAMPLE_ROWS = [
    {"task": "belebele", "model": "Llama-8B-FFT", "language": "swh_Latn", "n": "10",
     "score_type": "accuracy", "score": "0.6", "random_baseline": "0.25", "mean_latency_ms": "500"},
    {"task": "belebele", "model": "Llama-8B-FFT", "language": "luo_Latn", "n": "10",
     "score_type": "accuracy", "score": "0.3", "random_baseline": "0.25", "mean_latency_ms": "500"},
    {"task": "sib", "model": "Llama-8B-FFT", "language": "swh_Latn", "n": "10",
     "score_type": "accuracy", "score": "0.14", "random_baseline": "0.143", "mean_latency_ms": "300"},
    {"task": "flores", "model": "Llama-8B-FFT", "language": "eng_Latn-to-swh_Latn", "n": "10",
     "score_type": "corpus ChrF", "score": "39.2", "random_baseline": "", "mean_latency_ms": "1000"},
]


def test_coverage_matrix_groups_flores_direction_by_base_language():
    matrix = build_coverage_matrix(SAMPLE_ROWS, ["belebele", "sib", "flores"])
    assert ("flores", "swh_Latn") in matrix
    assert matrix[("flores", "swh_Latn")][0]["language"] == "eng_Latn-to-swh_Latn"


def test_render_html_marks_uncovered_languages_as_na():
    html_out = render_html(SAMPLE_ROWS)
    # Belebele has no Kikuyu/Kamba data in the sample rows -> should show n/a
    assert 'title="not covered by this dataset"' in html_out
    assert ">n/a<" in html_out


def test_render_html_includes_models_and_tasks():
    html_out = render_html(SAMPLE_ROWS)
    for needle in ["Llama-8B-FFT", "belebele", "sib", "flores"]:
        assert needle in html_out


def test_render_html_formats_percentage_vs_chrf_scale():
    html_out = render_html(SAMPLE_ROWS)
    assert "60.0%" in html_out   # belebele swahili: 0.6 -> 60.0%
    assert "39.2" in html_out    # flores chrF: already 0-100 scale


def test_build_report_writes_file(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SAMPLE_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(SAMPLE_ROWS)

    out_path = build_report([summary_path], tmp_path / "dashboard.html")
    assert out_path.exists()
    assert "LughaGen Evaluation Dashboard" in out_path.read_text()


def test_build_report_tolerates_missing_summary_file(tmp_path):
    out_path = build_report([tmp_path / "does_not_exist.csv"], tmp_path / "dashboard.html")
    assert out_path.exists()  # still writes an (empty) dashboard rather than crashing


def test_build_report_reads_round1_legacy_schema_directly(tmp_path):
    # Round 1's summary.csv used separate accuracy/chrf columns instead of
    # the new unified score/score_type -- the dashboard should render these
    # without anyone having to re-run the old evaluation first.
    legacy_path = tmp_path / "legacy_summary.csv"
    with legacy_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "model", "language", "n", "accuracy", "chrf", "mean_latency_ms"])
        writer.writeheader()
        writer.writerow({"task": "sib", "model": "Llama-8B-FFT", "language": "swh_Latn", "n": 204,
                          "accuracy": "0.275", "chrf": "", "mean_latency_ms": "1200"})
        writer.writerow({"task": "flores", "model": "Llama-8B-CPT", "language": "eng_Latn-to-swh_Latn", "n": 1012,
                          "accuracy": "", "chrf": "39.2", "mean_latency_ms": "2000"})

    out_path = build_report([legacy_path], tmp_path / "legacy_dashboard.html")
    content = out_path.read_text()
    assert "27.5%" in content  # 0.275 -> 27.5%
    assert "39.2" in content   # chrF stays on its own 0-100 scale

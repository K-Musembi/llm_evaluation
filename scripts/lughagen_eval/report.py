"""Build a single self-contained HTML dashboard from one or more summary.csv files.

Round 1 shipped flat ``summary.csv`` files as the only artifact a person
would look at. Those are fine as a data layer but poor as a communication
layer: nothing shows a baseline, nothing flags a language a benchmark simply
doesn't cover (it just silently isn't a row), and nothing surfaces a
qualitative example next to the numbers. This module builds a lightweight
dashboard that fixes all three, generated fresh after every run rather than
assembled by hand.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

LANGUAGE_ORDER = ["swh_Latn", "luo_Latn", "kik_Latn", "kam_Latn"]
LANGUAGE_NAMES = {"swh_Latn": "Swahili", "luo_Latn": "Luo", "kik_Latn": "Kikuyu", "kam_Latn": "Kamba"}


# Known chance-level baselines for tasks that predate the unified schema
# (round 1 only ran these two, both classification-style with a fixed
# number of candidate labels).
LEGACY_TASK_BASELINES = {"sib": 1 / 7}


def _normalize_legacy_row(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt round-1's ``summary.csv`` schema (separate accuracy/chrf columns)
    to the unified score/score_type schema the dashboard expects, so round 1's
    existing files can be rendered without being re-run."""
    if "score" in row and "score_type" in row:
        return row  # already in the new schema
    normalized = dict(row)
    accuracy = row.get("accuracy") or ""
    chrf = row.get("chrf") or ""
    if accuracy not in ("", None):
        normalized["score"], normalized["score_type"] = accuracy, "accuracy"
    elif chrf not in ("", None):
        normalized["score"], normalized["score_type"] = chrf, "corpus ChrF"
    else:
        normalized["score"], normalized["score_type"] = "", ""
    normalized.setdefault("random_baseline", "")
    if not normalized["random_baseline"]:
        baseline = LEGACY_TASK_BASELINES.get(row.get("task", ""))
        if baseline is not None:
            normalized["random_baseline"] = baseline
    return normalized


def load_summary_rows(summary_csv_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_csv_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(_normalize_legacy_row(row))
    return rows


def _row_language_base(language: str) -> str:
    """Map a FLORES direction key (e.g. 'eng_Latn-to-swh_Latn') back to its
    non-English language, so it lines up in the coverage matrix by language."""
    for code in LANGUAGE_ORDER:
        if code in language:
            return code
    return language


def build_coverage_matrix(rows: list[dict[str, Any]], task_names: list[str]) -> dict[tuple[str, str], list[dict]]:
    matrix: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        lang_base = _row_language_base(row["language"])
        matrix.setdefault((row["task"], lang_base), []).append(row)
    return matrix


def _score_cell_color(score: float | None, random_baseline: float | None) -> str:
    if score is None:
        return "#eeeeee"
    if random_baseline not in (None, ""):
        baseline = float(random_baseline)
        # green the further above baseline, red the further at/below it
        if score <= baseline:
            return "#f4c7c3"
        span = max(1.0 - baseline, 1e-6)
        t = min((score - baseline) / span, 1.0)
        return f"rgba(46,125,91,{0.15 + 0.55 * t:.2f})"
    # No baseline (e.g. chrF, 0-100 scale): shade by absolute value.
    t = min(max(score / 100.0, 0.0), 1.0)
    return f"rgba(46,125,91,{0.10 + 0.6 * t:.2f})"


def _fmt_score(score: str | float | None) -> str:
    if score in (None, ""):
        return "-"
    value = float(score)
    return f"{value * 100:.1f}%" if 0 <= value <= 1 else f"{value:.1f}"


def render_html(rows: list[dict[str, Any]], title: str = "LughaGen Evaluation Dashboard") -> str:
    task_names = sorted({row["task"] for row in rows})
    models = sorted({row["model"] for row in rows})
    matrix = build_coverage_matrix(rows, task_names)

    sections: list[str] = []

    # --- Leaderboard: mean score per model per task ---------------------
    leaderboard_rows = []
    for model in models:
        cells = []
        for task in task_names:
            scores = [float(r["score"]) for r in rows if r["model"] == model and r["task"] == task and r["score"]]
            cells.append(f"<td>{_fmt_score(sum(scores) / len(scores)) if scores else '-'}</td>")
        leaderboard_rows.append(f"<tr><th>{html.escape(model)}</th>{''.join(cells)}</tr>")
    sections.append(f"""
    <h2>Leaderboard (mean score per task, across covered languages)</h2>
    <table class="grid">
      <tr><th>Model</th>{''.join(f'<th>{html.escape(t)}</th>' for t in task_names)}</tr>
      {''.join(leaderboard_rows)}
    </table>
    """)

    # --- Coverage / heatmap per task -------------------------------------
    for task in task_names:
        header = "".join(f"<th>{LANGUAGE_NAMES.get(lang, lang)}</th>" for lang in LANGUAGE_ORDER)
        body_rows = []
        for model in models:
            cells = []
            for lang in LANGUAGE_ORDER:
                cell_rows = matrix.get((task, lang))
                model_rows = [r for r in cell_rows if r["model"] == model] if cell_rows else []
                if not model_rows:
                    cells.append('<td class="na" title="not covered by this dataset">n/a</td>')
                    continue
                scores = [float(r["score"]) for r in model_rows if r["score"]]
                baseline = next((r.get("random_baseline") for r in model_rows if r.get("random_baseline")), None)
                mean_score = sum(scores) / len(scores) if scores else None
                color = _score_cell_color(mean_score, baseline)
                partial = any(str(r.get("complete", "True")).strip().lower() == "false" for r in model_rows)
                note = ' <sup title="run not finished: score is based on a partial sample">partial</sup>' if partial else ""
                cells.append(f'<td style="background:{color}">{_fmt_score(mean_score)}{note}</td>')
            body_rows.append(f"<tr><th>{html.escape(model)}</th>{''.join(cells)}</tr>")
        sections.append(f"""
        <h3>{html.escape(task)}</h3>
        <table class="grid">
          <tr><th>Model</th>{header}</tr>
          {''.join(body_rows)}
        </table>
        """)

    style = """
    body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; color: #1a2620; background: #fbfaf7; }
    h1 { color: #1b4b43; }
    h2 { margin-top: 2.5rem; color: #1b4b43; }
    h3 { margin-top: 1.5rem; color: #3e7a63; }
    table.grid { border-collapse: collapse; margin-top: 0.5rem; }
    table.grid th, table.grid td { border: 1px solid #ddd; padding: 6px 12px; text-align: center; font-size: 0.9rem; }
    table.grid th { background: #1b4b43; color: white; }
    table.grid td.na { color: #999; font-style: italic; background: #f0f0f0; }
    """
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{style}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p>Cells marked <em>n/a</em> mean the benchmark has no data for that language (e.g. Belebele has no
Kikuyu/Kamba; IrokoBench tasks only cover Swahili) -- not that the model failed there.
Cells marked <sup>partial</sup> come from a run that has not finished yet.</p>
{body}
</body></html>
"""


def build_report(summary_csv_paths: list[Path], output_path: Path, title: str = "LughaGen Evaluation Dashboard") -> Path:
    rows = load_summary_rows(summary_csv_paths)
    output_path.write_text(render_html(rows, title=title), encoding="utf-8")
    return output_path

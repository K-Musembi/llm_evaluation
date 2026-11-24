"""Generalized evaluation runner.

This replaces the SIB/FLORES-specific ``evaluate_job``/``summarize`` pair in
the original script with versions written against the ``Task`` and
``ModelBackend`` interfaces, so every current and future benchmark shares one
code path for JSONL logging, resumability, human-review export, and
cross-task summarization.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .backends import BackendError, ModelBackend
from .budget import RunBudget
from .io_utils import append_jsonl, load_completed_indexes, read_ok_records, utc_now
from .tasks.base import Task
from .core_types import GenerationConfig, ModelOutput, ScoreResult


def job_language_key(task: Task, language: str) -> str:
    """The label used for a job's output filename and summary rows."""
    direction_key = getattr(task, "direction_key", None)
    return direction_key if direction_key is not None else language


class AbortRun(RuntimeError):
    """Raised when the backend keeps failing, so the run stops instead of
    filling the results folder with thousands of identical error rows."""


def evaluate_job(
    *, task: Task, model: str, language: str, output_dir: Path,
    backend: ModelBackend, config: GenerationConfig,
    limit: int | None = None, cache_dir: str | None = None,
    budget: RunBudget | None = None, max_consecutive_errors: int = 25,
) -> Path | None:
    """Run one (task, model, language) job, writing a resumable JSONL file.

    Returns ``None`` (and writes nothing) if the task doesn't cover this
    language -- e.g. Belebele has no Kikuyu, IrokoBench only has Swahili --
    so the gap is visible in logs rather than producing an empty/misleading
    output file.

    If ``budget`` expires mid-job the loop stops cleanly: everything done so far
    is on disk, the job's metrics are marked ``complete: false``, and the next
    run resumes from the first missing example. After ``max_consecutive_errors``
    backend failures in a row (CUDA out-of-memory, a dead endpoint, a model that
    won't load) it raises ``AbortRun`` rather than burning through the dataset.
    """
    supported = task.supported_languages()
    if supported is not None and language not in supported:
        print(f"[skip] {task.name}/{model}/{language}: not covered by this dataset "
              f"(supported: {sorted(supported)})")
        return None

    language_key = job_language_key(task, language)
    job_dir = output_dir / task.name / model
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / f"{language_key}.jsonl"
    completed = load_completed_indexes(output_path)

    interrupted = False
    had_errors = False
    consecutive_errors = 0
    abort_message: str | None = None

    for example in task.load_examples(language, limit, cache_dir):
        if example.index in completed:
            continue
        if budget is not None:
            budget.maybe_sync()
            if budget.expired():
                interrupted = True
                break
        prompt = task.build_prompt(example, language)
        record: dict[str, Any] = {
            "timestamp": utc_now(), "status": "ok", "task": task.name, "model": model,
            "language": language_key, "index": example.index, "prompt": prompt,
            "source": example.source, "target": example.target, "scoring_mode": task.scoring_mode,
        }
        try:
            if task.scoring_mode == "loglik_choice":
                choice_list = task.choices(example)
                if not choice_list:
                    raise BackendError(f"{task.name} is loglik_choice but returned no choices for this example.")
                output = backend.loglikelihood(model, prompt, choice_list, config)
            else:
                output = backend.generate(model, prompt, config)
            if not output.ok:
                raise BackendError(output.error or "backend returned an error with no message")
            score_result = task.score(example, output)
            record["raw_output"] = output.generated_text
            record["latency_ms"] = output.latency_ms
            record["choice_logprobs"] = output.choice_logprobs
            record["prediction"] = (output.generated_text or "").strip() if output.generated_text is not None else None
            record["normalized_prediction"] = score_result.normalized_prediction
            record["correct"] = score_result.correct
            record["item_score"] = score_result.item_score
            record["score_type"] = score_result.score_type
            consecutive_errors = 0
        except BackendError as error:
            record.update({"status": "error", "error": str(error)})
            had_errors = True
            consecutive_errors += 1
        append_jsonl(output_path, record)
        if consecutive_errors >= max_consecutive_errors:
            abort_message = (f"{task.name}/{model}/{language_key}: {consecutive_errors} consecutive backend "
                             f"errors; last: {record.get('error')}")
            break

    write_human_review(output_path, task, complete=not (interrupted or had_errors or abort_message))
    if budget is not None:
        budget.maybe_sync(force=True)
    if abort_message:
        raise AbortRun(abort_message)
    return output_path


def _score_result_from_record(record: dict[str, Any]) -> ScoreResult:
    return ScoreResult(
        item_score=record.get("item_score", 0.0) or 0.0,
        correct=record.get("correct"),
        normalized_prediction=record.get("normalized_prediction"),
        score_type=record.get("score_type", ""),
    )


def write_human_review(path: Path, task: Task, complete: bool = True) -> None:
    """Write a compact, spreadsheet-friendly view of one evaluation job."""
    job_records = read_ok_records(path)
    if not job_records:
        return
    review_path = path.with_suffix(".review.csv")
    metric_path = path.with_suffix(".metrics.json")
    fieldnames = [
        "index", "input", "reference_output", "model_output", "score_type", "item_score",
        "normalized_prediction", "latency_ms",
    ]
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in job_records:
            writer.writerow({
                "index": item["index"], "input": item["source"], "reference_output": item["target"],
                "model_output": item.get("prediction", ""), "score_type": item.get("score_type", ""),
                "item_score": item.get("item_score", ""), "normalized_prediction": item.get("normalized_prediction", ""),
                "latency_ms": item.get("latency_ms", ""),
            })

    results = [_score_result_from_record(r) for r in job_records]
    aggregate = task.aggregate(results)
    if task.name == "flores" and hasattr(task, "aggregate_corpus"):
        predictions = [item.get("prediction", "") or "" for item in job_records]
        references = [item["target"] for item in job_records]
        aggregate = {"score_type": "corpus ChrF", "score": task.aggregate_corpus(predictions, references)}

    metrics: dict[str, Any] = {
        "task": job_records[0]["task"], "model": job_records[0]["model"], "language": job_records[0]["language"],
        "examples": len(job_records), "complete": complete, "review_file": review_path.name, **aggregate,
    }
    metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(paths: Iterable[Path], tasks_by_name: dict[str, Task], output_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path is None:
            continue
        job_records = read_ok_records(path)
        if not job_records:
            continue
        task = tasks_by_name[job_records[0]["task"]]
        previous_complete = True
        metrics_file = path.with_suffix(".metrics.json")
        if metrics_file.exists():
            previous_complete = json.loads(metrics_file.read_text(encoding="utf-8")).get("complete", True)
        write_human_review(path, task, complete=previous_complete)
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        latencies = [item.get("latency_ms") for item in job_records]
        valid_latencies = [v for v in latencies if isinstance(v, (int, float))]
        rows.append({
            "task": metrics["task"], "model": metrics["model"], "language": metrics["language"],
            "n": metrics["examples"], "score_type": metrics.get("score_type", ""), "score": metrics.get("score", ""),
            "random_baseline": metrics.get("random_baseline", ""), "complete": metrics.get("complete", True),
            "mean_latency_ms": sum(valid_latencies) / len(valid_latencies) if valid_latencies else "",
        })

    summary_path = output_dir / "summary.csv"
    fieldnames = ["task", "model", "language", "n", "score_type", "score", "random_baseline", "complete",
                  "mean_latency_ms"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


@dataclass
class RunOutcome:
    paths: list[Path] = field(default_factory=list)
    stopped: str | None = None   # None = everything finished; "budget" or "aborted" otherwise
    message: str = ""


def run_jobs(
    task_jobs: dict[str, list[tuple[Task, str]]], models: list[str], *, backend: ModelBackend,
    config: GenerationConfig, output_dir: Path, limit: int | None = None, cache_dir: str | None = None,
    budget: RunBudget | None = None, max_consecutive_errors: int = 25,
) -> RunOutcome:
    """Run every (model, task, language) job, MODEL-OUTERMOST.

    Loading a checkpoint from Google Drive takes minutes, so each model is loaded
    once and runs all of its jobs before the next model is loaded. Jobs already
    finished in an earlier session cost nothing (no backend calls, so no model
    load); a partially finished job resumes from its first missing example.
    """
    outcome = RunOutcome()
    for model in models:
        for entries in task_jobs.values():
            for task, language in entries:
                if budget is not None and budget.expired():
                    outcome.stopped, outcome.message = "budget", "time budget reached; run again to resume"
                    return outcome
                try:
                    path = evaluate_job(task=task, model=model, language=language, limit=limit,
                                        output_dir=output_dir, backend=backend, config=config,
                                        cache_dir=cache_dir, budget=budget,
                                        max_consecutive_errors=max_consecutive_errors)
                except AbortRun as error:
                    outcome.stopped, outcome.message = "aborted", str(error)
                    return outcome
                if path is not None:
                    outcome.paths.append(path)
                if budget is not None and budget.hit:
                    outcome.stopped, outcome.message = "budget", "time budget reached; run again to resume"
                    return outcome
    return outcome


def expected_job_keys(task_jobs: dict[str, list[tuple[Task, str]]], models: list[str]) -> list[tuple[str, str, str]]:
    """(task, model, language_key) for every job that applies, skipping unsupported languages."""
    keys: list[tuple[str, str, str]] = []
    for model in models:
        for entries in task_jobs.values():
            for task, language in entries:
                supported = task.supported_languages()
                if supported is not None and language not in supported:
                    continue
                keys.append((task.name, model, job_language_key(task, language)))
    return keys


def job_status(output_dir: Path, expected: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """Per-job progress for planning sessions: complete / partial / not_started."""
    rows: list[dict[str, Any]] = []
    for task_name, model, language_key in expected:
        metrics_file = output_dir / task_name / model / f"{language_key}.metrics.json"
        if not metrics_file.exists():
            rows.append({"task": task_name, "model": model, "language": language_key, "examples": 0,
                         "status": "not_started"})
            continue
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append({"task": task_name, "model": model, "language": language_key,
                     "examples": metrics.get("examples", 0),
                     "status": "complete" if metrics.get("complete", True) else "partial"})
    return rows

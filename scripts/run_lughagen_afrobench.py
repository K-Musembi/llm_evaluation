#!/usr/bin/env python3
"""Evaluate LughaGen API models on AfroBench's SIB-200 and FLORES-200 tasks.

The runner deliberately calls the supplied ``/generate`` endpoint one prompt at
a time.  That makes the endpoint's per-request latency and complete response
metadata available in the JSONL audit trail, and makes interrupted jobs safely
resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
import sacrebleu
from datasets import load_dataset


DEFAULT_ENDPOINT = "https://picture-showers-exempt-foundation.trycloudflare.com/generate"
DEFAULT_MODELS = [
    "Llama-8B-FFT",
    "Llama-8B-CPT",
    "Llama-8B-QLoRA",
    "Gemma2-27B-FFT",
    "Gemma2-27B-CPT",
]
LANGUAGES = {
    "eng_Latn": "English",
    "swh_Latn": "Swahili",
    "luo_Latn": "Luo",
    "kik_Latn": "Kikuyu",
    "kam_Latn": "Kamba",
}
SIB_LABELS = [
    "science/technology", "travel", "politics", "sports", "health", "entertainment", "geography",
]


@dataclass(frozen=True)
class GenerationConfig:
    endpoint: str
    api_language: str
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    timeout: int
    retries: int


class EndpointError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def call_endpoint(model: str, prompt: str, config: GenerationConfig) -> dict[str, Any]:
    """Call the API with bounded retries and preserve the complete response."""
    payload = {
        "model": model,
        "prompt": prompt,
        "language": config.api_language,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "repetition_penalty": config.repetition_penalty,
    }
    last_error: Exception | None = None
    for attempt in range(config.retries + 1):
        try:
            response = requests.post(config.endpoint, json=payload, timeout=config.timeout)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("generated_text"), str):
                raise EndpointError("API response did not contain a string 'generated_text'.")
            return data
        except (requests.RequestException, ValueError, EndpointError) as error:
            last_error = error
            if attempt == config.retries:
                break
            time.sleep(min(2**attempt, 8))
    raise EndpointError(f"API request failed after {config.retries + 1} attempts: {last_error}")


def load_completed_indexes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if record.get("status") == "ok":
                    completed.add(int(record["index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sib_prompt(text: str, language: str) -> str:
    # AfroBench's SIB prompt 3, with the language name filled in.
    return (
        "You are an assistant able to classify topics in texts.\n\n"
        "Given the categories science/technology, travel, politics, sports, health, "
        "entertainment, or geography; what is the topic of the " + language +
        " statement below? Return only the category.\n\n"
        "text: " + text + "\ncategory:"
    )


def flores_prompt(source_text: str, source_language: str, target_language: str) -> str:
    # AfroBench's FLORES prompt 1, with the language names filled in.
    return (
        f"{source_language} sentence: {source_text}\n"
        f"{target_language} sentence: . Return only the translated sentence."
    )


def normalize_sib_response(response: str) -> str | None:
    normalized = response.casefold().strip()
    matches: list[tuple[int, str]] = []
    for label in SIB_LABELS:
        alternatives = [label]
        if label == "science/technology":
            alternatives += ["science and technology", "science technology"]
        for alternative in alternatives:
            position = normalized.find(alternative)
            if position >= 0:
                matches.append((position, label))
    return min(matches)[1] if matches else None


def strip_generation(text: str) -> str:
    """Remove only surrounding whitespace; prompts are not assumed to be echoed."""
    return text.strip()


def dataset_rows_sib(language_code: str, limit: int | None, cache_dir: str | None) -> Iterable[dict[str, Any]]:
    dataset = load_dataset("Davlan/sib200", language_code, split="test", cache_dir=cache_dir)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield {"index": index, "source": row["text"], "target": row["category"]}


def dataset_rows_flores(
    source_code: str, target_code: str, limit: int | None, cache_dir: str | None, dataset_name: str
) -> Iterable[dict[str, Any]]:
    # The current facebook/flores Hub dataset exposes each translation pair as
    # a configuration (for example ``swh_Latn-eng_Latn``), rather than a single
    # all-language table used by the original AfroBench API runner.
    pair_config = f"{source_code}-{target_code}"
    dataset = load_dataset(dataset_name, pair_config, split="devtest", cache_dir=cache_dir)
    source_column = f"sentence_{source_code}"
    target_column = f"sentence_{target_code}"
    missing = {source_column, target_column} - set(dataset.column_names)
    if missing:
        raise ValueError(f"FLORES config {pair_config!r} is missing expected columns: {sorted(missing)}")
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield {"index": index, "source": row[source_column], "target": row[target_column]}


def evaluate_job(
    *, task: str, model: str, source_code: str, target_code: str | None, limit: int | None,
    output_dir: Path, config: GenerationConfig, cache_dir: str | None, flores_dataset: str,
) -> Path:
    language_key = source_code if target_code is None else f"{source_code}-to-{target_code}"
    job_dir = output_dir / task / model
    job_dir.mkdir(parents=True, exist_ok=True)
    output_path = job_dir / f"{language_key}.jsonl"
    completed = load_completed_indexes(output_path)

    if task == "sib":
        rows = dataset_rows_sib(source_code, limit, cache_dir)
        source_name = LANGUAGES[source_code]
    else:
        assert target_code is not None
        rows = dataset_rows_flores(source_code, target_code, limit, cache_dir, flores_dataset)
        source_name, target_name = LANGUAGES[source_code], LANGUAGES[target_code]

    for row in rows:
        if row["index"] in completed:
            continue
        prompt = sib_prompt(row["source"], source_name) if task == "sib" else flores_prompt(
            row["source"], source_name, target_name
        )
        record: dict[str, Any] = {
            "timestamp": utc_now(), "status": "ok", "task": task, "model": model,
            "language": language_key, "index": row["index"], "prompt": prompt,
            "source": row["source"], "target": row["target"],
        }
        try:
            api_response = call_endpoint(model, prompt, config)
            raw_output = api_response["generated_text"]
            record["raw_output"] = raw_output
            record["prediction"] = strip_generation(raw_output)
            record["api_response"] = api_response
            if task == "sib":
                record["normalized_prediction"] = normalize_sib_response(raw_output)
                record["correct"] = record["normalized_prediction"] == row["target"]
        except EndpointError as error:
            record.update({"status": "error", "error": str(error)})
        append_jsonl(output_path, record)
    # Make a human-review file available as soon as this language/direction job
    # is complete; the aggregate summary is still generated at run completion.
    write_human_review(output_path)
    return output_path


def records(path: Path) -> list[dict[str, Any]]:
    # A stopped process can be resumed while its final write is still flushing.
    # Keep the most recent successful record per dataset index for scoring and
    # reviewer exports, while retaining every raw attempt in the JSONL audit.
    by_index: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                by_index[int(record["index"])] = record
    return [by_index[index] for index in sorted(by_index)]


def write_human_review(path: Path) -> None:
    """Write a compact, spreadsheet-friendly view of one evaluation job."""
    job_records = records(path)
    if not job_records:
        return
    first = job_records[0]
    task = first["task"]
    review_path = path.with_suffix(".review.csv")
    metric_path = path.with_suffix(".metrics.json")
    fieldnames = [
        "index", "input", "reference_output", "model_output", "score_type", "item_score",
        "normalized_prediction", "endpoint_latency_ms",
    ]
    review_rows: list[dict[str, Any]] = []
    for item in job_records:
        if task == "sib":
            score_type, item_score = "exact category accuracy", int(bool(item.get("correct")))
            normalized = item.get("normalized_prediction") or "INVALID"
        else:
            score_type = "sentence ChrF"
            item_score = sacrebleu.sentence_chrf(item["prediction"], [item["target"]]).score
            normalized = ""
        review_rows.append({
            "index": item["index"], "input": item["source"], "reference_output": item["target"],
            "model_output": item.get("prediction", ""), "score_type": score_type,
            "item_score": item_score, "normalized_prediction": normalized,
            "endpoint_latency_ms": item.get("api_response", {}).get("latency_ms", ""),
        })
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    metrics: dict[str, Any] = {
        "task": task, "model": first["model"], "language": first["language"], "examples": len(job_records),
        "review_file": review_path.name,
    }
    if task == "sib":
        metrics.update({
            "score_type": "accuracy", "score": sum(bool(item.get("correct")) for item in job_records) / len(job_records),
            "item_score_type": "exact category accuracy (0 or 1)",
        })
    else:
        metrics.update({
            "score_type": "corpus ChrF", "score": sacrebleu.corpus_chrf(
                [item["prediction"] for item in job_records], [[item["target"] for item in job_records]]
            ).score,
            "item_score_type": "sentence ChrF (for human inspection; headline score is corpus ChrF)",
        })
    metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(paths: Iterable[Path], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in paths:
        job_records = records(path)
        if not job_records:
            continue
        write_human_review(path)
        first = job_records[0]
        row = {"task": first["task"], "model": first["model"], "language": first["language"], "n": len(job_records)}
        if first["task"] == "sib":
            row["accuracy"] = sum(bool(item.get("correct")) for item in job_records) / len(job_records)
            row["chrf"] = ""
        else:
            predictions = [item["prediction"] for item in job_records]
            references = [item["target"] for item in job_records]
            row["accuracy"] = ""
            row["chrf"] = sacrebleu.corpus_chrf(predictions, [references]).score
        latencies = [item.get("api_response", {}).get("latency_ms") for item in job_records]
        valid_latencies = [latency for latency in latencies if isinstance(latency, (int, float))]
        row["mean_latency_ms"] = sum(valid_latencies) / len(valid_latencies) if valid_latencies else ""
        rows.append(row)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "model", "language", "n", "accuracy", "chrf", "mean_latency_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=["sib", "flores"], default=["sib", "flores"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--languages", nargs="+", choices=["swh_Latn", "luo_Latn", "kik_Latn", "kam_Latn"],
                        default=["swh_Latn", "luo_Latn", "kik_Latn", "kam_Latn"])
    parser.add_argument("--flores-directions", choices=["both", "to-english", "from-english"], default="both")
    parser.add_argument("--limit", type=int, help="Examples per task/language/direction; omit for full data.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/lughagen_afrobench"))
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory (for example /tmp/lughagen_hf_cache).")
    parser.add_argument("--flores-dataset", default="facebook/flores")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-language", default="Auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01, help="Must be > 0 for the supplied endpoint.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--summarize-existing", action="store_true",
                        help="Regenerate reviewer CSVs and summary.csv from existing JSONL files only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0; the supplied endpoint rejects zero.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_existing:
        paths = sorted(args.output_dir.glob("*/*/*.jsonl"))
        if not paths:
            raise SystemExit(f"No JSONL evaluation files found under {args.output_dir}")
        summarize(paths, args.output_dir)
        return 0
    config = GenerationConfig(
        endpoint=args.endpoint, api_language=args.api_language, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        timeout=args.timeout, retries=args.retries,
    )
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "created_at": utc_now(), "tasks": args.tasks, "models": args.models, "languages": args.languages,
        "limit": args.limit, "flores_directions": args.flores_directions, "flores_dataset": args.flores_dataset,
        "generation": asdict(config),
    }, indent=2) + "\n", encoding="utf-8")

    paths: list[Path] = []
    for task in args.tasks:
        for model in args.models:
            for language in args.languages:
                if task == "sib":
                    paths.append(evaluate_job(task=task, model=model, source_code=language, target_code=None,
                                              limit=args.limit, output_dir=args.output_dir, config=config,
                                              cache_dir=args.cache_dir, flores_dataset=args.flores_dataset))
                    continue
                directions = []
                if args.flores_directions in ("both", "to-english"):
                    directions.append((language, "eng_Latn"))
                if args.flores_directions in ("both", "from-english"):
                    directions.append(("eng_Latn", language))
                for source_code, target_code in directions:
                    paths.append(evaluate_job(task=task, model=model, source_code=source_code, target_code=target_code,
                                              limit=args.limit, output_dir=args.output_dir, config=config,
                                              cache_dir=args.cache_dir, flores_dataset=args.flores_dataset))
    summarize(paths, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

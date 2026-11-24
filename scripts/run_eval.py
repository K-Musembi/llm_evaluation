#!/usr/bin/env python3
"""Evaluate LughaGen models across SIB-200, FLORES-200, Belebele, and IrokoBench.

Generalized replacement for ``run_lughagen_afrobench.py``. Same JSONL audit
trail / resumability / review-CSV behavior, extended to:

* more tasks, added as plugins under ``lughagen_eval/tasks/`` (see
  ``lughagen_eval/tasks/base.py`` for the interface new tasks implement);
* a choice of backend -- the existing HTTP ``/generate`` endpoint, or a
  model run directly on the host machine, via ``--backend {api,local}``.

Examples
--------
Run everything against the API, as before::

    python scripts/run_eval.py --tasks sib flores --backend api \\
        --endpoint https://.../generate

Round 2, comprehension/knowledge first, log-likelihood scored, small sample
while iterating::

    python scripts/run_eval.py --tasks belebele afrimmlu afrixnli \\
        --backend local --model-paths-file models.json --limit 20

On Google Colab (models on Drive, 5.5 h per session, results synced to Drive)::

    python scripts/run_eval.py --backend local --model-paths-file models.json \\
        --quantization 4bit --tasks belebele afrimmlu afrixnli flores sib \\
        --output-dir /content/results/round2 \\
        --sync-dir /content/drive/MyDrive/lughagen/results/round2 \\
        --max-runtime-hours 5.5

Run the same command again in the next session to resume. Check progress or
validate the setup without running anything::

    python scripts/run_eval.py ... --status
    python scripts/run_eval.py ... --check-setup

Regenerate summary.csv and the dashboard from existing JSONL files only,
without calling any backend::

    python scripts/run_eval.py --summarize-existing --output-dir results/round2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from lughagen_eval.backends import APIBackend, LocalBackend, ModelBackend
from lughagen_eval.budget import RunBudget, restore_tree, sync_tree
from lughagen_eval.io_utils import utc_now
from lughagen_eval.model_specs import check_model_specs, estimate_vram_gb, parse_model_specs, weights_size_gb
from lughagen_eval.report import build_report
from lughagen_eval.runner import expected_job_keys, job_status, run_jobs, summarize
from lughagen_eval.tasks import ALL_TASK_NAMES, DIRECTIONAL_TASKS, SINGLE_INSTANCE_TASKS, Task
from lughagen_eval.core_types import GenerationConfig

DEFAULT_ENDPOINT = "https://picture-showers-exempt-foundation.trycloudflare.com/generate"
DEFAULT_MODELS = ["Llama-8B-FFT", "Llama-8B-CPT", "Llama-8B-QLoRA", "Gemma2-27B-FFT", "Gemma2-27B-CPT"]
ALL_LANGUAGES = ["swh_Latn", "luo_Latn", "kik_Latn", "kam_Latn"]


def load_model_paths(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_backend(args: argparse.Namespace) -> ModelBackend:
    if args.backend == "api":
        return APIBackend(endpoint=args.endpoint, score_endpoint=args.score_endpoint)
    if args.model_paths_file is None:
        raise SystemExit("--backend local requires --model-paths-file (a JSON {model_name: path} map).")
    return LocalBackend(model_paths=load_model_paths(args.model_paths_file), device=args.device,
                        quantization=getattr(args, "quantization", "none"), dtype=getattr(args, "dtype", "float16"))


def build_task_instances(task_names: list[str], languages: list[str], flores_directions: str,
                          flores_dataset: str) -> dict[str, list[tuple[Task, str]]]:
    """Return {task_name: [(task_instance, language), ...]} to run.

    FLORES expands into one task instance per (direction, target language)
    pair; every other task uses one shared instance across all languages.
    """
    jobs: dict[str, list[tuple[Task, str]]] = {}
    for name in task_names:
        if name in SINGLE_INSTANCE_TASKS:
            task = SINGLE_INSTANCE_TASKS[name]()
            jobs[name] = [(task, language) for language in languages]
        elif name in DIRECTIONAL_TASKS:
            flores_cls = DIRECTIONAL_TASKS[name]
            entries: list[tuple[Task, str]] = []
            for language in languages:
                if flores_directions in ("both", "to-english"):
                    entries.append((flores_cls(language, "eng_Latn", flores_dataset), language))
                if flores_directions in ("both", "from-english"):
                    entries.append((flores_cls("eng_Latn", language, flores_dataset), language))
            jobs[name] = entries
        else:
            raise SystemExit(f"Unknown task {name!r}. Choose from: {ALL_TASK_NAMES}")
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASK_NAMES, default=["sib", "flores"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Default: the keys of --model-paths-file (local) or the five round-1 model names (api).")
    parser.add_argument("--languages", nargs="+", choices=ALL_LANGUAGES, default=ALL_LANGUAGES)
    parser.add_argument("--flores-directions", choices=["both", "to-english", "from-english"], default="both")
    parser.add_argument("--flores-dataset", default="facebook/flores")
    parser.add_argument("--limit", type=int, help="Examples per job; omit for full data.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/lughagen_eval"))
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory.")

    parser.add_argument("--backend", choices=["api", "local"], default="api")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="API backend: /generate endpoint URL.")
    parser.add_argument("--score-endpoint", default=None,
                        help="API backend: optional logprob endpoint, required for loglik_choice tasks over the API.")
    parser.add_argument("--model-paths-file", help="Local backend: JSON {model_name: local_checkpoint_path} map.")
    parser.add_argument("--device", default="cuda", help="Local backend: torch device string.")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none",
                        help="Local backend: load weights in 8/4-bit so larger models fit a small GPU (e.g. a 15 GB T4). "
                             "Slightly changes scores vs full precision; it is recorded in run_config.json.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "auto"], default="float16",
                        help="Local backend: compute dtype (a T4 has no bfloat16 support).")

    parser.add_argument("--max-runtime-hours", type=float, default=None,
                        help="Stop cleanly after this long (checked between examples); re-run to resume.")
    parser.add_argument("--sync-dir", type=Path, default=None,
                        help="Copy results here (e.g. a Google Drive folder) during the run, and restore from it at start.")
    parser.add_argument("--sync-every-minutes", type=float, default=10.0)
    parser.add_argument("--max-consecutive-errors", type=int, default=25,
                        help="Abort the run after this many backend failures in a row (e.g. GPU out-of-memory).")
    parser.add_argument("--status", action="store_true",
                        help="Show which jobs are complete/partial/not started, then exit.")
    parser.add_argument("--check-setup", action="store_true",
                        help="Validate model folders and estimate GPU memory needs, then exit.")

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--api-language", default="Auto")

    parser.add_argument("--summarize-existing", action="store_true",
                        help="Regenerate review CSVs, summary.csv, and the dashboard from existing JSONL only.")
    parser.add_argument("--report-only", action="store_true",
                        help="Just rebuild dashboard.html from an existing summary.csv (no JSONL rescan).")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def all_task_registry() -> dict[str, Task]:
    """One task instance per known task name, for summarizing results already on disk.

    FLORES's aggregate does not depend on direction, so any pair works here.
    """
    registry: dict[str, Task] = {name: cls() for name, cls in SINGLE_INSTANCE_TASKS.items()}
    for name, cls in DIRECTIONAL_TASKS.items():
        registry[name] = cls("swh_Latn", "eng_Latn")
    return registry


def resolve_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        return args.models
    if args.backend == "local" and args.model_paths_file:
        return list(load_model_paths(args.model_paths_file))
    return DEFAULT_MODELS


def gpu_info() -> tuple[str, float] | None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.name, props.total_memory / 1e9
    except ImportError:
        pass
    return None


def print_check_setup(args: argparse.Namespace) -> int:
    if args.backend != "local" or not args.model_paths_file:
        print("--check-setup applies to --backend local with --model-paths-file.")
        return 1
    specs = parse_model_specs(load_model_paths(args.model_paths_file))
    problems = check_model_specs(specs, args.models)
    gpu = gpu_info()
    print(f"GPU: {gpu[0]} ({gpu[1]:.1f} GB)" if gpu else "GPU: none detected (Colab: Runtime > Change runtime type > GPU)")
    print(f"Quantization requested: {args.quantization}\n")
    for name in args.models:
        spec = specs.get(name)
        if spec is None:
            continue
        size = weights_size_gb(spec.base_path if spec.is_adapter and spec.base_path else spec.path)
        if size == 0.0:
            continue
        need = estimate_vram_gb(size)
        verdict = ""
        if gpu:
            fits = [mode for mode in ("none", "8bit", "4bit") if need[mode] <= gpu[1]]
            verdict = f"  -> fits on this GPU with: {', '.join(fits) if fits else 'NOTHING (needs a bigger GPU)'}"
        print(f"{name}: {size:.1f} GB of weights; est. GPU need  fp16 {need['none']:.1f} / 8bit {need['8bit']:.1f} "
              f"/ 4bit {need['4bit']:.1f} GB{verdict}")
    print("(estimates assume a 16-bit checkpoint; if it was saved in fp32, halve them)")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("\nSetup looks OK." if not problems else f"\n{len(problems)} problem(s) found.")
    return 1 if problems else 0


def print_status(output_dir: Path, task_jobs, models: list[str]) -> None:
    rows = job_status(output_dir, expected_job_keys(task_jobs, models))
    counts = {"complete": 0, "partial": 0, "not_started": 0}
    for row in rows:
        counts[row["status"]] += 1
        print(f"{row['status']:<12} {row['model']:<18} {row['task']:<9} {row['language']:<24} n={row['examples']}")
    print(f"\n{counts['complete']} complete, {counts['partial']} partial, {counts['not_started']} not started "
          f"(of {len(rows)} jobs)")


def write_run_config(args: argparse.Namespace, config: GenerationConfig) -> None:
    """Record settings; keep a history of sessions since one run spans several."""
    path = args.output_dir / "run_config.json"
    previous: dict = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    session = {"started_at": utc_now(), "models": args.models, "tasks": args.tasks,
               "max_runtime_hours": args.max_runtime_hours}
    path.write_text(json.dumps({
        "created_at": previous.get("created_at", utc_now()), "tasks": args.tasks, "models": args.models,
        "languages": args.languages, "limit": args.limit, "backend": args.backend,
        "quantization": args.quantization if args.backend == "local" else None,
        "dtype": args.dtype if args.backend == "local" else None,
        "flores_directions": args.flores_directions, "flores_dataset": args.flores_dataset,
        "generation": asdict(config), "sessions": previous.get("sessions", []) + [session],
    }, indent=2) + "\n", encoding="utf-8")


def write_outputs(output_dir: Path) -> Path:
    """Re-summarize EVERY job on disk (not just this session's) and rebuild the dashboard."""
    paths = sorted(output_dir.glob("*/*/*.jsonl"))
    summary_path = summarize(paths, all_task_registry(), output_dir)
    dashboard_path = build_report([summary_path], output_dir / "dashboard.html")
    print(f"Wrote {summary_path}\nWrote {dashboard_path}")
    return summary_path


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.max_runtime_hours is not None and args.max_runtime_hours <= 0:
        raise SystemExit("--max-runtime-hours must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        dashboard_path = build_report([args.output_dir / "summary.csv"], args.output_dir / "dashboard.html")
        print(f"Wrote {dashboard_path}")
        return 0

    args.models = resolve_models(args)
    task_jobs = build_task_instances(args.tasks, args.languages, args.flores_directions, args.flores_dataset)

    if args.check_setup:
        return print_check_setup(args)

    if args.sync_dir is not None:
        restored = restore_tree(args.sync_dir, args.output_dir)
        print(f"Restored {restored} file(s) from {args.sync_dir}" if restored else f"Nothing to restore from {args.sync_dir}")

    if args.status:
        print_status(args.output_dir, task_jobs, args.models)
        return 0

    if args.summarize_existing:
        if not sorted(args.output_dir.glob("*/*/*.jsonl")):
            raise SystemExit(f"No JSONL evaluation files found under {args.output_dir}")
        write_outputs(args.output_dir)
        return 0

    if args.backend == "api" and args.temperature <= 0:
        raise SystemExit("--temperature must be > 0 for the API backend; it rejects zero.")

    backend = build_backend(args)
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, timeout=args.timeout, retries=args.retries,
        api_language=args.api_language,
    )
    write_run_config(args, config)

    sync_fn = (lambda: sync_tree(args.output_dir, args.sync_dir)) if args.sync_dir is not None else None
    budget = RunBudget(
        max_seconds=args.max_runtime_hours * 3600 if args.max_runtime_hours else None,
        sync_fn=sync_fn, sync_interval_seconds=args.sync_every_minutes * 60,
    )
    outcome = run_jobs(task_jobs, args.models, backend=backend, config=config, output_dir=args.output_dir,
                       limit=args.limit, cache_dir=args.cache_dir, budget=budget,
                       max_consecutive_errors=args.max_consecutive_errors)

    write_outputs(args.output_dir)
    if args.sync_dir is not None:
        sync_tree(args.output_dir, args.sync_dir)
        print(f"Synced results to {args.sync_dir}")

    print(f"Session time: {budget.elapsed() / 3600:.2f} h")
    if outcome.stopped == "budget":
        print("STOPPED: time budget reached. Everything so far is saved. Run the same command again to resume.")
        return 10
    if outcome.stopped == "aborted":
        print(f"ABORTED: {outcome.message}\nFix the cause (e.g. restart the runtime after a GPU out-of-memory), "
              "then run the same command again to resume.")
        return 20
    unfinished = [row for row in job_status(args.output_dir, expected_job_keys(task_jobs, args.models))
                  if row["status"] != "complete"]
    if unfinished:
        print(f"FINISHED WITH GAPS: {len(unfinished)} job(s) still have failed or missing examples "
              f"(first: {unfinished[0]['model']}/{unfinished[0]['task']}/{unfinished[0]['language']}). "
              "Check the 'error' rows in their .jsonl files; running the same command again retries only those examples.")
        return 30
    print("All requested jobs finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

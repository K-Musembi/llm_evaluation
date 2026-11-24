#!/usr/bin/env python3
"""Push evaluation results to GitHub (run this from Colab after, or between, sessions).

Copies the small result files (summary.csv, dashboard.html, run_config.json,
per-job *.metrics.json and *.review.csv) into <repo>/results/<run-name>/,
commits, and pushes. Raw *.jsonl logs stay on Google Drive unless you pass
--include-jsonl.

The token is read from an environment variable (default GITHUB_TOKEN), never
from the command line, so it does not appear in shell history or notebook output.

    python scripts/publish_results.py --results-dir /content/results/round2 \\
        --repo-dir /content/llm_evaluation --run-name lughagen_round2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from lughagen_eval.publish import PublishError, publish_results, select_publishable_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path("/content/llm_evaluation"))
    parser.add_argument("--run-name", required=True, help="Folder name under results/ in the repo.")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--include-jsonl", action="store_true", help="Also publish raw per-example JSONL logs.")
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--message", default=None)
    parser.add_argument("--no-push", action="store_true", help="Commit locally but do not push.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be published; change nothing.")
    args = parser.parse_args()

    if args.dry_run:
        selected, skipped = select_publishable_files(args.results_dir, args.include_jsonl, args.max_file_mb)
        print(f"Would publish {len(selected)} file(s) to results/{args.run_name}/")
        for path in selected:
            print(f"  + {path.relative_to(args.results_dir)}")
        for path, reason in skipped:
            print(f"  - {path.relative_to(args.results_dir)}  ({reason})")
        return 0

    token = os.environ.get(args.token_env)
    if not token and not args.no_push:
        print(f"Note: ${args.token_env} is not set. The push only works for a public repo you have "
              f"credentials for; in Colab add a GITHUB_TOKEN secret (see README).", file=sys.stderr)
    try:
        outcome = publish_results(args.results_dir, args.repo_dir, args.run_name, token=token, branch=args.branch,
                                  include_jsonl=args.include_jsonl, max_file_mb=args.max_file_mb,
                                  message=args.message, push=not args.no_push)
    except PublishError as error:
        print(f"Publish failed: {error}", file=sys.stderr)
        return 1
    if not outcome["committed"]:
        print("Nothing new to publish (results/%s is already up to date)." % args.run_name)
    else:
        print(f"Committed {outcome['files']} file(s)" + (" and pushed." if outcome["pushed"] else " (not pushed)."))
    for path, reason in outcome["skipped"]:
        print(f"  skipped {path.name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

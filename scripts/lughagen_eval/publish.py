"""Publish evaluation results from a Colab session to GitHub.

What goes to GitHub: the small, human-readable files -- ``summary.csv``,
``dashboard.html``, ``run_config.json``, each job's ``*.metrics.json`` and
``*.review.csv``. What stays on Google Drive: the raw per-example ``*.jsonl``
audit logs (they grow into tens of MB, and GitHub rejects any file over
100 MB). ``include_jsonl=True`` opts them in, still skipping any file bigger
than ``max_file_mb``.

Authentication uses a personal access token passed only on the individual git
commands that need it. It is never written into ``.git/config`` and is scrubbed
from any error text, so it cannot leak into notebook output.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

RESULT_SUFFIXES = (".csv", ".json", ".html", ".log", ".md")
GIT_IDENTITY = ["-c", "user.name=LughaGen Colab Runner", "-c", "user.email=colab-runner@users.noreply.github.com"]


class PublishError(RuntimeError):
    pass


def select_publishable_files(results_dir: Path, include_jsonl: bool = False,
                             max_file_mb: float = 50.0) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Split a results folder into (files to publish, [(skipped file, reason)])."""
    selected: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    limit = max_file_mb * 1024 * 1024
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file() or any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl" and not include_jsonl:
            skipped.append((path, "raw per-example log stays on Drive (pass include_jsonl to publish it)"))
        elif suffix != ".jsonl" and suffix not in RESULT_SUFFIXES:
            skipped.append((path, "not a results file"))
        elif path.stat().st_size > limit:
            skipped.append((path, f"larger than {max_file_mb:g} MB"))
        else:
            selected.append(path)
    return selected, skipped


def _scrub(text: str, token: str | None) -> str:
    return text.replace(token, "***") if token else text


def run_git(args: list[str], cwd: Path, token: str | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr.strip() or result.stdout.strip() or "no output")
        # the command line itself can contain the authenticated URL, so scrub both
        raise PublishError(_scrub(f"git {' '.join(args)} failed: {detail}", token))
    return result.stdout


def clean_url(url: str) -> str:
    """Remove any embedded credentials and normalise ssh-style GitHub URLs to https."""
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:"):]
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return url  # a local path or other remote: leave untouched
    host = parts.hostname + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def authenticated_url(url: str, token: str | None) -> str:
    base = clean_url(url)
    if not token or not base.startswith("https://"):
        return base
    parts = urlsplit(base)
    return urlunsplit((parts.scheme, f"x-access-token:{quote(token, safe='')}@{parts.netloc}", parts.path, "", ""))


def clone_or_update(repo_url: str, dest: Path, token: str | None = None, branch: str = "main") -> Path:
    """Clone the repo into ``dest`` (or fast-forward an existing clone) without leaving the token behind."""
    dest = Path(dest)
    auth = authenticated_url(repo_url, token)
    if (dest / ".git").exists():
        run_git(["pull", "--rebase", auth, branch], cwd=dest, token=token)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", "--branch", branch, auth, str(dest)], cwd=dest.parent, token=token)
        run_git(["remote", "set-url", "origin", clean_url(repo_url)], cwd=dest)
    return dest


def publish_results(results_dir: Path, repo_dir: Path, run_name: str, *, token: str | None = None,
                    branch: str = "main", include_jsonl: bool = False, max_file_mb: float = 50.0,
                    remote_url: str | None = None, message: str | None = None, push: bool = True) -> dict:
    """Copy results into ``<repo>/results/<run_name>/``, commit, and push."""
    results_dir, repo_dir = Path(results_dir), Path(repo_dir)
    if not run_name or "/" in run_name or run_name.startswith("."):
        raise PublishError(f"run_name must be a simple folder name, got {run_name!r}")
    selected, skipped = select_publishable_files(results_dir, include_jsonl, max_file_mb)
    if not selected:
        raise PublishError(f"No publishable result files found under {results_dir}")

    destination = repo_dir / "results" / run_name
    for source in selected:
        target = destination / source.relative_to(results_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    relative = f"results/{run_name}"
    run_git(["add", "--", relative], cwd=repo_dir)
    if not run_git(["status", "--porcelain", "--", relative], cwd=repo_dir).strip():
        return {"committed": False, "pushed": False, "files": len(selected), "skipped": skipped}
    run_git([*GIT_IDENTITY, "commit", "-m", message or f"Add evaluation results: {run_name}"], cwd=repo_dir)

    pushed = False
    if push:
        origin = remote_url or run_git(["remote", "get-url", "origin"], cwd=repo_dir).strip()
        auth = authenticated_url(origin, token)
        run_git(["pull", "--rebase", auth, branch], cwd=repo_dir, token=token)  # pick up anything pushed meanwhile
        run_git(["push", auth, f"HEAD:{branch}"], cwd=repo_dir, token=token)
        pushed = True
    return {"committed": True, "pushed": pushed, "files": len(selected), "skipped": skipped}

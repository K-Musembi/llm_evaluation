"""Session time budget and Google Drive syncing, for Colab runs.

A Colab session can end at any moment (usage limits, idle timeout, a closed
tab), and its local disk disappears with it. Two protections:

* ``RunBudget`` -- a soft wall-clock limit (e.g. 5.5 h). The runner checks it
  between examples and stops cleanly, so results are saved and synced *before*
  Colab cuts the session. Re-running the same command resumes where it stopped.
* ``sync_tree`` / ``restore_tree`` -- results are written to fast local disk
  (appending line-by-line to the Drive mount is slow and can lose buffered
  writes) and copied to Drive periodically; at the start of the next session
  they are copied back so resumability works across sessions.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable


class RunBudget:
    def __init__(self, max_seconds: float | None = None, sync_fn: Callable[[], object] | None = None,
                 sync_interval_seconds: float = 600.0, clock: Callable[[], float] = time.monotonic):
        self.max_seconds = max_seconds
        self.sync_fn = sync_fn
        self.sync_interval_seconds = sync_interval_seconds
        self._clock = clock
        self._start = clock()
        self._last_sync = self._start
        self.hit = False

    def elapsed(self) -> float:
        return self._clock() - self._start

    def expired(self) -> bool:
        if self.max_seconds is not None and self.elapsed() >= self.max_seconds:
            self.hit = True
        return self.hit

    def maybe_sync(self, force: bool = False) -> bool:
        """Run the sync callback if it is due (or ``force``). Returns True if it ran."""
        if self.sync_fn is None:
            return False
        now = self._clock()
        if force or now - self._last_sync >= self.sync_interval_seconds:
            self.sync_fn()
            self._last_sync = now
            return True
        return False


def _complete_lines_only(data: bytes) -> bytes:
    """Cut a JSONL byte string back to its last full line (drop a half-written tail)."""
    return data[: data.rfind(b"\n") + 1]


def sync_tree(src: Path, dst: Path) -> int:
    """Copy new/changed files from ``src`` to ``dst``. Returns the number copied.

    JSONL files are copied only up to their last complete line, so a snapshot
    taken while the evaluator is mid-write never contains a half-written record.
    """
    copied = 0
    if not src.is_dir():
        return 0
    for source_file in src.rglob("*"):
        if not source_file.is_file():
            continue
        target = dst / source_file.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_file.suffix == ".jsonl":
            data = _complete_lines_only(source_file.read_bytes())
            if not target.exists() or target.stat().st_size != len(data):
                target.write_bytes(data)
                copied += 1
        elif (not target.exists() or target.stat().st_size != source_file.stat().st_size
              or source_file.stat().st_mtime > target.stat().st_mtime + 1):
            shutil.copy2(source_file, target)
            copied += 1
    return copied


def restore_tree(src: Path, dst: Path) -> int:
    """Bring results back from Drive into a fresh local folder at session start.

    Never overwrites newer local work: a file is copied only if it is missing
    locally, or (for JSONL) the Drive copy has strictly more data. CSV/JSON
    summaries are regenerated from the JSONL, so they are not restored over
    existing files.
    """
    copied = 0
    if not src.is_dir():
        return 0
    for source_file in src.rglob("*"):
        if not source_file.is_file():
            continue
        target = dst / source_file.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or (source_file.suffix == ".jsonl"
                                    and source_file.stat().st_size > target.stat().st_size):
            shutil.copy2(source_file, target)
            copied += 1
    return copied

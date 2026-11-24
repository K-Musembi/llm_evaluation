from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_indexes(path: Path) -> set[int]:
    """Indexes already successfully evaluated, for resuming an interrupted run."""
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


def read_ok_records(path: Path) -> list[dict[str, Any]]:
    """The most recent successful record per index (a resumed run may retry)."""
    by_index: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                by_index[int(record["index"])] = record
    return [by_index[i] for i in sorted(by_index)]

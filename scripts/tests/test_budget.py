import os
import time

from lughagen_eval.budget import RunBudget, restore_tree, sync_tree


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_budget_without_limit_never_expires():
    clock = FakeClock()
    budget = RunBudget(max_seconds=None, clock=clock)
    clock.t = 10**9
    assert not budget.expired() and not budget.hit


def test_budget_expires_at_limit_and_stays_expired():
    clock = FakeClock()
    budget = RunBudget(max_seconds=100, clock=clock)
    clock.t = 99
    assert not budget.expired()
    clock.t = 100
    assert budget.expired() and budget.hit
    clock.t = 0  # even if the clock were odd afterwards, an expired budget stays expired
    assert budget.expired()


def test_maybe_sync_respects_interval_and_force():
    clock = FakeClock()
    calls = []
    budget = RunBudget(sync_fn=lambda: calls.append(clock.t), sync_interval_seconds=600, clock=clock)
    clock.t = 599
    assert not budget.maybe_sync() and calls == []
    clock.t = 600
    assert budget.maybe_sync() and calls == [600]
    clock.t = 650
    assert not budget.maybe_sync()          # interval restarts after each sync
    assert budget.maybe_sync(force=True)    # force ignores the interval
    assert len(calls) == 2


def test_maybe_sync_without_sync_fn_is_a_noop():
    assert RunBudget().maybe_sync(force=True) is False


def test_sync_tree_copies_new_and_changed_files(tmp_path):
    src, dst = tmp_path / "local", tmp_path / "drive"
    (src / "task" / "model").mkdir(parents=True)
    (src / "summary.csv").write_text("a,b\n1,2\n")
    (src / "task" / "model" / "x.metrics.json").write_text("{}")
    assert sync_tree(src, dst) == 2
    assert (dst / "task" / "model" / "x.metrics.json").exists()
    assert sync_tree(src, dst) == 0  # nothing changed -> nothing copied
    (src / "summary.csv").write_text("a,b\n1,2\n3,4\n")
    assert sync_tree(src, dst) == 1


def test_sync_tree_never_copies_a_half_written_jsonl_line(tmp_path):
    src, dst = tmp_path / "local", tmp_path / "drive"
    src.mkdir()
    (src / "job.jsonl").write_bytes(b'{"index": 0}\n{"index": 1}\n{"index": 2, "tru')  # mid-write snapshot
    sync_tree(src, dst)
    assert (dst / "job.jsonl").read_bytes() == b'{"index": 0}\n{"index": 1}\n'
    # once the writer finishes the line, the next sync picks it up
    with (src / "job.jsonl").open("ab") as handle:
        handle.write(b'ncated"}\n')
    sync_tree(src, dst)
    assert (dst / "job.jsonl").read_bytes().count(b"\n") == 3


def test_sync_tree_missing_source_is_harmless(tmp_path):
    assert sync_tree(tmp_path / "nope", tmp_path / "drive") == 0


def test_restore_copies_missing_files_into_a_fresh_session(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    drive.mkdir()
    (drive / "job.jsonl").write_bytes(b'{"index": 0}\n')
    local.mkdir()
    assert restore_tree(drive, local) == 1
    assert (local / "job.jsonl").read_bytes() == b'{"index": 0}\n'


def test_restore_never_overwrites_newer_local_work(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    drive.mkdir()
    local.mkdir()
    (drive / "job.jsonl").write_bytes(b'{"index": 0}\n')
    (local / "job.jsonl").write_bytes(b'{"index": 0}\n{"index": 1}\n')   # local is ahead
    (drive / "summary.csv").write_text("old")
    (local / "summary.csv").write_text("new")
    assert restore_tree(drive, local) == 0
    assert (local / "job.jsonl").read_bytes().count(b"\n") == 2
    assert (local / "summary.csv").read_text() == "new"


def test_restore_prefers_a_longer_drive_jsonl(tmp_path):
    drive, local = tmp_path / "drive", tmp_path / "local"
    drive.mkdir()
    local.mkdir()
    (drive / "job.jsonl").write_bytes(b'{"index": 0}\n{"index": 1}\n')
    (local / "job.jsonl").write_bytes(b'{"index": 0}\n')
    assert restore_tree(drive, local) == 1
    assert (local / "job.jsonl").read_bytes().count(b"\n") == 2

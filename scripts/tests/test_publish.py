import subprocess

import pytest

from lughagen_eval.publish import (
    PublishError, authenticated_url, clean_url, clone_or_update, publish_results, run_git,
    select_publishable_files,
)

IDENT = ["-c", "user.name=t", "-c", "user.email=t@example.com"]


def git(cwd, *args):
    subprocess.run(["git", *IDENT, *args], cwd=cwd, check=True, capture_output=True, text=True)


def make_remote_and_clone(tmp_path):
    """A bare 'GitHub' with one commit, plus a working clone of it."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-b", "main", str(seed)], check=True, capture_output=True)
    (seed / "README.md").write_text("repo")
    git(seed, "add", ".")
    git(seed, "commit", "-m", "init")
    git(seed, "push", str(bare), "main")
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "--branch", "main", str(bare), str(work)], check=True, capture_output=True)
    return bare, work


def make_results(folder):
    (folder / "sib" / "M").mkdir(parents=True)
    (folder / "summary.csv").write_text("task,model\nsib,M\n")
    (folder / "dashboard.html").write_text("<html></html>")
    (folder / "run_config.json").write_text("{}")
    (folder / "sib" / "M" / "swh_Latn.jsonl").write_text('{"index": 0}\n')
    (folder / "sib" / "M" / "swh_Latn.review.csv").write_text("index\n0\n")
    (folder / "sib" / "M" / "swh_Latn.metrics.json").write_text("{}")
    (folder / "weights.bin").write_bytes(b"0")
    return folder


def test_selection_excludes_raw_jsonl_by_default(tmp_path):
    results = make_results(tmp_path / "res")
    selected, skipped = select_publishable_files(results)
    names = {p.name for p in selected}
    assert {"summary.csv", "dashboard.html", "run_config.json", "swh_Latn.review.csv", "swh_Latn.metrics.json"} <= names
    assert "swh_Latn.jsonl" not in names
    reasons = {p.name: r for p, r in skipped}
    assert "stays on Drive" in reasons["swh_Latn.jsonl"]
    assert "weights.bin" in reasons  # unknown file types are never published


def test_selection_can_include_jsonl_but_respects_the_size_cap(tmp_path):
    results = make_results(tmp_path / "res")
    (results / "sib" / "M" / "big.jsonl").write_bytes(b"x" * (2 * 1024 * 1024))
    selected, skipped = select_publishable_files(results, include_jsonl=True, max_file_mb=1)
    names = {p.name for p in selected}
    assert "swh_Latn.jsonl" in names and "big.jsonl" not in names
    assert any(p.name == "big.jsonl" and "larger than" in r for p, r in skipped)


def test_clean_and_authenticated_urls():
    assert clean_url("https://user:pw@github.com/o/r.git") == "https://github.com/o/r.git"
    assert clean_url("git@github.com:o/r.git") == "https://github.com/o/r.git"
    assert clean_url("/some/local/path.git") == "/some/local/path.git"
    assert authenticated_url("https://github.com/o/r.git", "tok") == "https://x-access-token:tok@github.com/o/r.git"
    assert authenticated_url("https://github.com/o/r.git", None) == "https://github.com/o/r.git"
    assert authenticated_url("/local/remote.git", "tok") == "/local/remote.git"  # can't inject into a path
    assert "a%2Fb" in authenticated_url("https://github.com/o/r.git", "a/b")      # token is URL-encoded


def test_token_is_scrubbed_from_git_errors(tmp_path):
    with pytest.raises(PublishError) as excinfo:
        run_git(["SECRET-TOKEN-123"], cwd=tmp_path, token="SECRET-TOKEN-123")
    assert "SECRET-TOKEN-123" not in str(excinfo.value)
    assert "***" in str(excinfo.value)


def test_publish_commits_and_pushes_only_the_small_files(tmp_path):
    bare, work = make_remote_and_clone(tmp_path)
    results = make_results(tmp_path / "res")
    outcome = publish_results(results, work, "run1", remote_url=str(bare))
    assert outcome["committed"] and outcome["pushed"]

    check = tmp_path / "check"
    subprocess.run(["git", "clone", str(bare), str(check)], check=True, capture_output=True)
    published = {p.name for p in (check / "results" / "run1").rglob("*") if p.is_file()}
    assert "summary.csv" in published and "swh_Latn.review.csv" in published
    assert "swh_Latn.jsonl" not in published and "weights.bin" not in published


def test_publishing_twice_with_no_changes_is_a_noop(tmp_path):
    bare, work = make_remote_and_clone(tmp_path)
    results = make_results(tmp_path / "res")
    publish_results(results, work, "run1", remote_url=str(bare))
    again = publish_results(results, work, "run1", remote_url=str(bare))
    assert again["committed"] is False and again["pushed"] is False


def test_publish_integrates_a_commit_pushed_in_the_meantime(tmp_path):
    bare, work = make_remote_and_clone(tmp_path)
    # someone else (e.g. you, from your laptop) pushes while the Colab session was running
    other = tmp_path / "other"
    subprocess.run(["git", "clone", str(bare), str(other)], check=True, capture_output=True)
    (other / "notes.md").write_text("code change")
    git(other, "add", ".")
    git(other, "commit", "-m", "laptop change")
    git(other, "push", "origin", "main")

    outcome = publish_results(make_results(tmp_path / "res"), work, "run1", remote_url=str(bare))
    assert outcome["pushed"]
    check = tmp_path / "check"
    subprocess.run(["git", "clone", str(bare), str(check)], check=True, capture_output=True)
    assert (check / "notes.md").exists() and (check / "results" / "run1" / "summary.csv").exists()


def test_publish_no_push_commits_locally_only(tmp_path):
    bare, work = make_remote_and_clone(tmp_path)
    outcome = publish_results(make_results(tmp_path / "res"), work, "run1", remote_url=str(bare), push=False)
    assert outcome["committed"] and not outcome["pushed"]
    log = subprocess.run(["git", "log", "--oneline"], cwd=bare, capture_output=True, text=True).stdout
    assert "evaluation results" not in log


def test_publish_rejects_bad_run_names_and_empty_results(tmp_path):
    bare, work = make_remote_and_clone(tmp_path)
    results = make_results(tmp_path / "res")
    for bad in ["", "a/b", ".hidden"]:
        with pytest.raises(PublishError):
            publish_results(results, work, bad, remote_url=str(bare))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PublishError, match="No publishable"):
        publish_results(empty, work, "run1", remote_url=str(bare))


def test_clone_or_update_leaves_no_credentials_in_git_config(tmp_path):
    bare, _ = make_remote_and_clone(tmp_path)
    dest = tmp_path / "fresh"
    clone_or_update(str(bare), dest, token="irrelevant-for-local-path")
    config = (dest / ".git" / "config").read_text()
    assert "irrelevant-for-local-path" not in config
    # a second call updates the existing clone instead of failing
    clone_or_update(str(bare), dest)

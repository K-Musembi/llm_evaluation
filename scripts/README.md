# LughaGen AfroBench runner

`run_lughagen_afrobench.py` evaluates the supplied LughaGen `/generate` API on
the AfroBench SIB-200 prompt and FLORES-200 translation prompt. It writes every
request, complete API response, prediction, reference, and metric input to
resumable JSONL files, followed by `summary.csv`.

Run commands from the evaluation workspace with the project virtualenv.

## Smoke test (20 SIB-200 examples)

```bash
venv/bin/python scripts/run_lughagen_afrobench.py \
  --tasks sib --models Llama-8B-FFT --languages swh_Latn \
  --limit 20 --cache-dir /tmp/lughagen_hf_cache \
  --output-dir results/lughagen_smoke
```

## Full SIB-200 run

```bash
venv/bin/python scripts/run_lughagen_afrobench.py \
  --tasks sib --cache-dir /tmp/lughagen_hf_cache \
  --output-dir results/lughagen_sib_full
```

## FLORES-200

AfroBench uses `facebook/flores`, which currently requires Hugging Face access
approval and authentication. The current Hub dataset exposes each direction as
a language-pair configuration; the runner selects it automatically. After
accepting its dataset conditions and setting `HF_TOKEN`, run both
English↔target directions with:

```bash
HF_TOKEN=... venv/bin/python scripts/run_lughagen_afrobench.py \
  --tasks flores --cache-dir /tmp/lughagen_hf_cache \
  --output-dir results/lughagen_flores_full
```

The supplied endpoint rejects `temperature=0`; the runner uses `0.01` (the
lowest accepted near-greedy setting) and records it in `run_config.json`.

---

# Round 2: generalized evaluation framework

`run_lughagen_afrobench.py` above is kept as-is for continuity with round 1.
Round 2 adds two new benchmarks (Belebele, IrokoBench) and the ability to run
against a model directly instead of only through the API. Both live in a new
package, `lughagen_eval/`, and a new entry point, `run_eval.py`, so nothing
about the original script had to change.

```
scripts/
  lughagen_eval/
    core_types.py     Example / ModelOutput / ScoreResult / GenerationConfig
    backends.py        APIBackend, LocalBackend, FakeBackend (ModelBackend interface)
    local_scoring.py   pure-Python log-likelihood math (no torch required to import)
    io_utils.py        JSONL audit trail: append, resume, read
    model_specs.py     models.json parsing, adapter detection, GPU-fit estimates
    budget.py          session time budget + Drive sync/restore
    publish.py         copy small results into the repo, commit, push (token-safe)
    runner.py          evaluate_job / write_human_review / summarize (task-agnostic)
    report.py          dashboard.html generator
    tasks/
      base.py           the Task interface every benchmark implements
      sib.py            SIB-200 (migrated from run_lughagen_afrobench.py)
      flores.py         FLORES-200 (migrated)
      belebele.py        Belebele (new)
      irokobench.py       AfriMMLU / AfriXNLI / AfriMGSM (new)
  run_eval.py           CLI: --tasks, --backend {api,local}, --max-runtime-hours, --sync-dir, --status, --check-setup
  publish_results.py    CLI: push results to GitHub
  (repo root) notebooks/colab_eval.ipynb, notebooks/models.example.json
  tests/                pytest suite, no network/GPU required (see "Testing" below)
```

## Why a `Task` interface

Adding Belebele and IrokoBench by copy-pasting SIB/FLORES's branches into the
original script would have made the file's `if task == "sib": ... elif task
== "flores": ...` branching grow linearly with every new benchmark. Instead,
every benchmark is a class implementing `lughagen_eval/tasks/base.py`'s
`Task`: `load_examples`, `build_prompt`, `score`, `aggregate`, and (for
multiple-choice tasks) `choices`. `runner.evaluate_job` is written only
against this interface, so it didn't need to change to add the two new
benchmarks, and won't need to change for the next one either.

Each task declares a `scoring_mode`:

| scoring_mode        | used by                          | how it's scored                                                        |
|----------------------|-----------------------------------|--------------------------------------------------------------------------|
| `generate_match`     | `sib`, `flores`                   | generate free text, compare/measure it directly (exact-match / chrF)     |
| `generate_extract`   | `afrimgsm`                         | generate free text, pull the final number out, compare to the reference |
| `loglik_choice`      | `belebele`, `afrimmlu`, `afrixnli` | score each fixed candidate continuation by log-likelihood, take the max  |

`loglik_choice` is the fairer mode for a model that hasn't been heavily
instruction-tuned yet (see the "why loglik_choice" note in `tasks/belebele.py`
and the round-1→round-2 discussion this implements) -- it removes "did the
model format its answer exactly as asked" as a source of error separate from
"does the model actually know the answer."

## Language coverage gaps -- read this before running round 2

Neither new benchmark covers all four of this project's languages:

* **Belebele**: Swahili and Luo only. No Kikuyu, no Kamba.
* **IrokoBench** (AfriMMLU / AfriXNLI / AfriMGSM): **Swahili only.**

`Task.supported_languages()` encodes this per task. `evaluate_job` checks it
before doing any work: an unsupported language/task pair is skipped, logged
to stdout as `[skip] ...`, and produces no output file -- rather than being
silently absent the way a hand-maintained results table might make it look
like a language was tested and scored zero. `dashboard.html` (see below)
renders these as an explicit `n/a` cell rather than leaving them blank.

If you later want Kikuyu/Kamba coverage on Belebele specifically, the
dataset's own card suggests machine-translating the English questions/answers
onto the FLORES-200 passages those languages already have (see
`tasks/belebele.py`'s module docstring) -- treat that as a deliberate,
flagged addition, not a default.

## Running on Google Colab (models on Drive)

This is the intended way to run round 2. Everything is driven from
`notebooks/colab_eval.ipynb`; the notes below explain what it does and why.

### The idea in one paragraph

Colab sessions are temporary: the local disk vanishes when a session ends, the GPU you get is not
guaranteed, and usage limits and idle timeouts are dynamic (up to 12 h at best). So a run is designed to be
**split across sessions and resumable**. Results are written to fast local disk, copied to your Drive every
10 minutes, the session stops itself cleanly after `MAX_RUNTIME_HOURS` (default 5.5), and the next session
restores from Drive and continues from the first missing example. Finished jobs cost nothing on re-run (no
requests, so the model isn't even loaded).

### One-time setup

1. **Drive layout.** Keep models and results under one folder, e.g.

   ```
   MyDrive/lughagen/
     models.json              <- copy of notebooks/models.example.json with your real paths
     models/LughaGen-Gemma2-27B-SFT-V2/ ...
     models/LughaGen-Llamma-8B-SFT-V2.1/ ... <- each checkpoint has config.json + weights
     results/                 <- created automatically
   ```

   The example JSON uses the two model folder names supplied for this run, including the `Llamma` spelling.
   A QLoRA/LoRA checkpoint that is only an adapter (it has `adapter_config.json`) is supported: its base model
   is read from that file, or set `base_model_path` in `models.json`.
2. **GitHub token** (lets Colab pull the code and push results). GitHub ▸ Settings ▸ Developer settings ▸
   Personal access tokens ▸ *Fine-grained tokens* ▸ Generate. Restrict it to **only this repository** with
   **Contents: Read and write** (menu names may differ slightly). Copy it once.
3. **Colab Secrets** (🔑 in the left sidebar), each with *Notebook access* switched on:
   `GITHUB_TOKEN` (the token above) and `HF_TOKEN` (Hugging Face token; accept the FLORES dataset terms first).
   Secrets keep the tokens out of the notebook text, so the notebook is safe to commit.
4. Open the notebook in Colab (`File ▸ Open notebook ▸ GitHub`), set the runtime to **GPU**, run all.

### The routine for every session

`Runtime ▸ Run all`. The notebook checks the GPU, mounts Drive, fetches the latest code, validates your models
(`--check-setup`), shows progress (`--status`), runs until done or out of time, and then you run the publish
cell. Repeat until `--status` shows every job `complete`.

| exit code | meaning | what to do |
|---|---|---|
| 0 | everything requested is finished | publish and finish |
| 10 | time budget reached, all progress saved | run all again in a new session |
| 20 | aborted: backend failed 25 times in a row (typically GPU out of memory) | restart the runtime, run all again |
| 30 | finished, but some examples failed | run all again; only those are retried |

### What goes to GitHub and what stays on Drive

| | where | why |
|---|---|---|
| raw per-example logs `*.jsonl`, in-progress state | **Drive** (`results/<run>/`) | they grow to tens of MB; GitHub rejects files over 100 MB |
| `summary.csv`, `dashboard.html`, `run_config.json`, per-job `*.metrics.json` and `*.review.csv` | **GitHub** (`results/<run>/`) | small, readable, diff-able; this is what you share |

`scripts/publish_results.py` does the copy, commit and push (the notebook's last cells call it). Use `--dry-run`
to preview, `--include-jsonl` to also publish raw logs (files over 50 MB are still skipped), `--no-push` to commit
only. It runs `git pull --rebase` first, so a commit you pushed from your laptop in the meantime is merged, not
overwritten. The token is passed only to the single git command that needs it: it is not stored in `.git/config`
and is scrubbed from error messages. After a session, `git pull` on your own machine brings the results down.

To get this code onto GitHub in the first place (once, from your laptop):

```bash
cd ~/lughagen/llm_evaluation
tar -xzf ~/Downloads/colab_eval_framework.tar.gz     # unpacks scripts/ and notebooks/
cd scripts && python -m pytest tests -q && cd ..      # expect: 160 passed
git add scripts notebooks && git commit -m "Colab evaluation framework" && git push
```

### GPU memory: read this before the first run

A free Colab GPU is *typically* a 15 GB T4, but it is not guaranteed; the notebook prints what you got.
`--check-setup` estimates each model's need from its weight files on Drive and says which loading modes fit.

* Llama-8B in 16-bit needs about 16 GB, so on a T4 use `--quantization 4bit` (about 5 GB) or `8bit` (about 9 GB).
* **Gemma2-27B is unlikely to fit a free T4 even in 4-bit** (roughly 14-16 GB before activations). Treat
  `--check-setup`'s verdict as authoritative. Options: run it on a larger GPU (Colab Pro's L4/A100), leave it out
  of `MODELS` for now, or evaluate it through the HTTP API as in round 1.
* Quantization shifts scores slightly compared with 16-bit. It is recorded in `run_config.json`; evaluate every model
  under the **same** setting so the comparison stays fair, and don't compare a 4-bit run with a 16-bit one.
* T4s have no bfloat16, so the default compute type is `float16`.

### Planning sessions

Run the notebook once with `LIMIT = 5` (a few examples per job) and look at `mean_latency_ms` in `summary.csv`:
multiply by the example counts (FLORES: 1012 sentences x 8 directions; SIB-200: 204 x 4 languages; Belebele: 900 per
language) to estimate hours per model. Model loading from Drive adds several minutes per model per session and is
not interruptible, which is why the default budget (5.5 h) leaves a margin. Models run one at a time, in the order
of `--models`, and each is loaded once per session, so to spread work across sessions you only need to rerun the
same command.

## Running round 2 (other setups: API, or a GPU server you control)

Against the API, exactly as before, plus the two new benchmarks:

```bash
venv/bin/python scripts/run_eval.py \
  --tasks belebele afrimmlu afrixnli afrimgsm flores sib \
  --backend api --endpoint https://.../generate \
  --cache-dir /tmp/lughagen_hf_cache \
  --output-dir results/lughagen_round2
```

**`loglik_choice` tasks (Belebele, AfriMMLU, AfriXNLI) need a way to get
per-continuation log-probabilities.** The API backend only has this if the
LughaGen server exposes a scoring endpoint; point `--score-endpoint` at it:

```bash
  --score-endpoint https://.../score
```

If there is no such endpoint yet, running these three tasks with
`--backend api` will fail immediately with a clear `BackendError` telling you
to add one or switch to `--backend local`. That's deliberate -- silently
falling back to "generate the letter and hope the format is right" would
reintroduce exactly the format-compliance confound this mode exists to avoid.

**Directly on the GPU host**, once the model's weights are on that machine:

```bash
pip install torch transformers --break-system-packages   # once, on that host
cat > models.json <<'JSON'
{
  "LughaGen-Gemma2-27B-SFT-V2": "/models/LughaGen-Gemma2-27B-SFT-V2",
  "LughaGen-Llamma-8B-SFT-V2.1": "/models/LughaGen-Llamma-8B-SFT-V2.1"
}
JSON
venv/bin/python scripts/run_eval.py \
  --tasks belebele afrimmlu afrixnli \
  --backend local --model-paths-file models.json --device cuda \
  --output-dir results/lughagen_round2
```

`LocalBackend` computes true per-token log-likelihoods via `transformers`, so
`loglik_choice` tasks work out of the box here with no server changes needed.
It also generates text for `generate_match`/`generate_extract` tasks the same
way the API does, so `--tasks sib flores` works unchanged with
`--backend local` too.

**Recommended round-2 order** (comprehension/knowledge first while those are
scored the fairer log-likelihood way; translation and classification, the two
weakest spots in round 1, last, once you have independent evidence of whether
the model "knows" the languages at all):

```bash
--tasks belebele afrimmlu afrixnli afrimgsm flores sib
```

## Regenerating the dashboard / summary only

```bash
venv/bin/python scripts/run_eval.py --summarize-existing --output-dir results/lughagen_round2
# or, from an existing summary.csv only, including round 1's:
venv/bin/python scripts/run_eval.py --report-only --output-dir results/lughagen_sib_full
```

`report.py` also reads round 1's original `summary.csv` schema directly (it
detects and normalizes the old `accuracy`/`chrf` columns), so you can build a
dashboard from the existing round-1 files without re-running anything:

```python
from lughagen_eval.report import build_report
build_report(
    [Path("results/lughagen_sib_full/summary.csv"), Path("results/lughagen_flores_full/summary.csv")],
    Path("results/round1_dashboard.html"),
)
```

The dashboard adds three things flat CSVs don't give you: a leaderboard
normalized against each task's random-guess baseline where one exists, a
model × language heatmap per task, and an explicit coverage matrix so a gap
like "no Kikuyu on Belebele" is a visible `n/a` cell instead of a silent
absence.

## Adding a new benchmark

1. Add `tasks/your_task.py` implementing `Task` (see `tasks/belebele.py` for
   a `loglik_choice` example, `tasks/sib.py` for `generate_match`,
   `tasks/irokobench.py`'s `AfriMgsmTask` for `generate_extract`).
2. Register it in `tasks/__init__.py`'s `SINGLE_INSTANCE_TASKS` (or
   `DIRECTIONAL_TASKS` if, like FLORES, it needs one instance per direction
   rather than per language).
3. Add unit tests mirroring `tests/test_tasks_belebele.py`: prompt shape,
   scoring on a few hand-written examples, and the language-coverage gap if
   any. No network access is needed -- every task's HF dataset loader is a
   small standalone function (`_load_raw_rows` / `_load_afrimmlu_rows` /
   etc.) that tests monkeypatch out.

## Testing

The whole suite runs without network access, a GPU, or `torch`/`transformers`
installed -- every dataset loader and both real backends are exercised
through dependency injection (`monkeypatch`) or the network-free `FakeBackend`
instead:

```bash
pip install pytest sacrebleu requests datasets --break-system-packages
cd scripts
python -m pytest tests/ -v
```

At the time of writing this is 160 tests covering: the pure log-likelihood
math (`local_scoring`), both real backends' HTTP/retry/error behavior
(`APIBackend`, mocked `requests`), each task's prompt building and scoring in
isolation, the runner end-to-end (including resumability after a simulated
mid-run failure, and the Kikuyu/Kamba/IrokoBench-Swahili-only coverage gaps),
the dashboard builder (including reading round 1's legacy schema), and the
CLI's own helper functions.

**What isn't, and can't be, tested here:** `LocalBackend` against a real
model, and either backend against a real LughaGen endpoint -- both need a GPU
or a live server this sandbox doesn't have. To check those for real:

1. **API backend, smoke-sized:** run the "Smoke test" style command above
   with `--backend api --limit 5` against a real endpoint and read the
   resulting `*.jsonl`/`*.review.csv` by hand -- five requests is enough to
   confirm payload shape, retries, and logging all look right before
   committing to a full run.
2. **Local backend:** on the GPU host, after `pip install torch
   transformers`, run:
   ```bash
   python -c "
   from lughagen_eval.backends import LocalBackend
   from lughagen_eval.types import GenerationConfig
   b = LocalBackend({'test-model': '/path/to/a/small/checkpoint'})
   print(b.generate('test-model', 'The capital of Kenya is', GenerationConfig(max_new_tokens=10)))
   print(b.loglikelihood('test-model', 'Q: 2+2=?\nA:', [' 3', ' 4', ' 5'], GenerationConfig()))
   "
   ```
   The `loglikelihood` call should print four (or however many) log-prob
   floats with the correct answer's the least negative -- confirm that by
   hand on one or two known examples before trusting it across a full run.
3. **`--score-endpoint`:** if/when the LughaGen server adds one, smoke-test
   it the same way as (1) with a two-or-three-continuation request and check
   the returned `logprobs` list is the same length and in the same order as
   the `continuations` you sent.

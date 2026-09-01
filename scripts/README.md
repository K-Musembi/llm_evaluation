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

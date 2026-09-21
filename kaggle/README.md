# Kaggle connection (Component 1: the cloud trainer)

The human runs a **Python notebook** in the Kaggle web UI — no CLI, no dataset
uploads. The C2 eval environment lives in **this repo** and is imported by the
notebook via a `git clone`; trained/measured artifacts travel back here.

```mermaid
flowchart LR
  A["this repo (C2: grid, data, metrics, layaenv)"] -->|"public on GitHub"| B["clone @ pinned commit"]
  B --> C["Kaggle notebook (C1)"]
  C -->|"pip install laya"| D["Hugging Face: convaiinnovations/laya + subcheckpoints"]
  C --> E["run layaenv grid -> /kaggle/working/report.jsonl + probe + sample preds"]
  E --> F["laya-artifacts.zip"]
  F -->|"human downloads"| G["agent validates vs frozen schema, gates (C3)"]
```

## The rule

The notebook is the only Kaggle surface. It clones this repo at a pinned
commit, so a gate re-checks the exact `layaenv` code that produced the
numbers. The agent never runs the Kaggle CLI, never launches/uploades a
notebook, and never gates on anything a cloud run didn't return.

## Runtime

- Accelerator **GPU T4 x2**, **Internet ON** — required (weights + eval data
  are pulled from HF inside the notebook).
- The `layaenv` package is torch-free on its own; it only imports `laya`/`torch`
  inside the cloud run and on this device for local regression runs.

## Notebook #1: v0-baseline (unblocks P0-7)

What it does:

1. Clones this repo at `REF` and imports `layaenv` from the clone.
2. `pip install laya` (pulls models from Hugging Face on first `laya.load`).
3. Runs `configs/eval_grid.yaml` against all four variants —
   `english` (`convaiinnovations/laya`), `multilingual` (`…/multilingual`),
   `typed_decisions` (`…/typed-decisions`), and `router` (`laya.Router`).
4. Emits `report.jsonl` (frozen schema), `probe.jsonl` (raw SDK output on the
   probe call, so the adapter can be tightened against the real API), and
   `summary.md` + `run_meta.json`, zipped to `laya-artifacts.zip`.

Grid cells (v0-baseline-r1): typed-decisions (gold + teacher dists),
Banking77 (cardinality), SST-5 (ordinal proxy), XNLI en + **XNLI Khmer**
(OOD stress). Local cells in `configs/eval_grid.yaml` activate the moment you
drop your private sets into `evaldata/`.

## Human workflow

1. Open Kaggle → New Notebook → File → Import Notebook →
   `kaggle/notebooks/v0_baseline.ipynb`.
2. Settings: Accelerator `GPU T4 x2`, Internet `On`, Language `Python`.
3. Run All (~15–30 min on T4 x2 for the default caps).
4. Download `laya-artifacts.zip` (Data → Output) and hand it back.

## Agent workflow (offline, C3 on this device)

1. Validate every `report.jsonl` row against `layaenv/schema.py` (schema v1.0).
2. Review `probe.jsonl` and tighten `layaenv/backends.py` extraction if the
   SDK returns keys the adapter doesn't know.
3. Gate the owning ticket only from returned numbers.
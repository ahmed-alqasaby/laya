# Plan: three components, every ticket mapped

This repo's unit of work splits into exactly three components. A ticket is
assigned to the component where its primary work physically happens; tickets
that span components are tagged with the full chain (e.g. `C1→C2→C3`).

## The three components

### C1 — Cloud trainer (Kaggle notebook)

Pulls base models from **Hugging Face**, loads the datasets, and *trains*.
Runs on `GPU T4 x2`, Internet ON, in a single Python notebook the human runs
in the Kaggle web UI.

Outputs (written to `/kaggle/working`, downloaded by the human):

- trained checkpoints (adapter weights + merge config)
- `report.jsonl` rows in the frozen schema (emitted by the imported C2 harness)
- `telemetry.jsonl` when a ticket asks for it

### C2 — Evaluation environment (this repo)

Everything that *measures*, lives **here** and is imported into the notebook:

- frozen eval grid config (`configs/eval_grid.yaml`, versioned + tagged)
- eval datasets: cardinality buckets, state-length buckets, domain sets,
  novelty split, multi-domain corpus, visual data mix (manifests committed)
- metrics suite + report schema (one command → one report artifact)
- gate definitions (the number each exit gate requires)
- router / SDK code the harness calls

The notebook clones this repo at a pinned commit and runs the C2 harness
against the model it just trained. C2 is also runnable on this device for
regression checks.

### C3 — Local device (this repo, this machine)

After the human hands the artifacts back, all *validation, inference, and
gating* happen here:

- validate every `report.jsonl` / `telemetry.jsonl` row against the frozen schema
- move the trained model + all needed files (config, tokenizer, adapter,
  report rows) **back into this repo**
- run local inference sanity + the C2 metrics suite against the returned model
- gate the owning ticket from the returned numbers
- export / quantize / serve / telemetry-loop work runs here on the device

## Data flow

```
[repo/C2] frozen grid, buckets, splits, metrics, router/SDK
    │  git push (public repo)
    ▼
[C1] human runs notebook (Kaggle, T4x2, Internet ON)
    │  clones repo @ pinned commit  →  imports C2 harness
    │  pulls base model from Hugging Face
    │  trains  →  runs C2 eval  →  writes report.jsonl + ckpts + telemetry.jsonl
    │  zips to laya-artifacts.zip; human downloads
    ▼
[C3] this device
    │  validate rows vs frozen schema
    │  move model + needed files back into repo
    │  local inference + metrics (C2) → regression/gate decisions → gh
```

Hard rule: the only source of gate numbers is a cloud-trained model run
through C2. No fake models, no offline smoke gates, no scaffold standing in
for a real run.

## Ticket map (all 52 open issues)

### P0 — foundation (C2, then the first real run)

| Fx | Ticket | Comp | Where the work physically happens |
|----|--------|------|-----------------------------------|
| #1 | P0-1 stub eval grid schema | C2 | frozen versioned config checked into repo |
| #2 | P0-2 cardinality buckets | C2 | eval data in repo |
| #3 | P0-3 state-length buckets | C2 | eval data in repo |
| #4 | P0-4 domain sets | C2 | eval data + provenance in repo |
| #5 | P0-5 novelty split | C2 | held-out split in repo + dedup report |
| #6 | P0-6 metrics suite | C2 | frozen one-command metrics in repo, imported by notebook |
| #7 | P0-7 baseline report | C1→C2→C3 | notebook pulls current HF checkpoints, runs C2 harness → report handed back, validated + tagged v0-baseline here |

### P1.1 — calibration + abstention

| Fx | Ticket | Comp | Where |
|----|--------|------|-------|
| #8 | P1.1-1 training-time calibration | C1 | loss baked into training in notebook; ECE via C2 harness |
| #9 | P1.1-2 in-loop temperature | C1 | temperature fit inside training loop; per-bucket post-fit in C2 |
| #10 | P1.1-3 act/escalate rejector | C1 | train rejector on cloud; AUROC + risk-coverage via C2 |
| #11 | P1.1-4 OOD detector | C1→C3 | trained on cloud; stress-test suite committed + run here |
| #12 | P1.1-5 distill soft uncertainty (opt) | C1 | teacher→student on cloud |
| #13 | P1.1-GATE | C3 | gate on returned numbers vs frozen schema |

### P1.2 — cardinality + context

| Fx | Ticket | Comp | Where |
|----|--------|------|-------|
| #14 | P1.2-1 unlock 8k context | C1 | config + Flash Attention on cloud GPUs; latency/memory to C2 schema |
| #15 | P1.2-2 4k milestone | C1 | measurement run on cloud, report row added |
| #16 | P1.2-3 8k milestone | C1 | same, at 8k |
| #17 | P1.2-4 32k stretch | C1 | continued training + eval |
| #18 | P1.2-5 retrieval option head | C1 | architecture + training on cloud |
| #19 | P1.2-6 two-stage training | C1 | contrastive pre-train → RLCD on cloud |
| #20 | P1.2-7 permutation augmentation | C1 | same training run as #19 |
| #21 | P1.2-8 flat-accuracy curves | C2 | two plots in C2 report schema from cloud runs |
| #22 | P1.2-9 re-calibrate per bucket | C1→C3 | refit on cloud; ECE ≤ 0.075 verified here |
| #23 | P1.2-GATE | C3 | gate |

### P1.4 — soft distributions

| Fx | Ticket | Comp | Where |
|----|--------|------|-------|
| #24 | P1.4-1 diagnostics | C1→C3 | cloud ablations; diagnosis report committed here |
| #25 | P1.4-2 intervention | C1 | implement + train on cloud; before/after in report |
| #26 | P1.4-3 teacher-matching gate | C1→C2 | carve ambiguous subset; reported as its own row |
| #27 | P1.4-GATE | C3 | gate |

### P1.3 — score primitive

| Fx | Ticket | Comp | Where |
|----|--------|------|-------|
| #28 | P1.3-1 ordinal data | C2 | ≥300 ordinal examples/domain + rubrics committed in repo |
| #29 | P1.3-2 head adjustments | C1 | train on cloud; justified per #24 |
| #30 | P1.3-3 distill (opt) | C1 | teacher-distill on cloud |
| #31 | P1.3-GATE | C3 | gate |

### P1.5 — generalization

| Fx | Ticket | Comp | Where |
|----|--------|------|-------|
| #32 | P1.5-1 multi-domain corpus | C2 | corpus + manifest committed in repo |
| #33 | P1.5-2 novelty split | C2 | strict split + dedup report |
| #34 | P1.5-3 train + zero-shot eval | C1→C2 | train on cloud; eval on C2 split; learning curve in report |
| #35 | P1.5-4 calibration regression | C3 | re-run C2 metrics here against returned model |
| #36 | P1.5-GATE | C3 | gate |

### P1.6 — router + telemetry

| Fx | Ticket | Comp | Where |
|----|--------|------|-------|
| #37 | P1.6-1 heuristic router | C2→C3 | router code in repo; dispatch test run here |
| #38 | P1.6-2 telemetry logging | C3 | logs from local traffic; schema matches P0-6 |
| #39 | P1.6-3 telemetry threshold | C3 | volume/coverage measured + documented here |
| #40 | P1.6-4 learned router policy | C3 | train on logged telemetry here; shadow vs heuristic |

### Phase 1 exit

| Fx | Ticket | Comp | Where |
|----|--------|------|-------|
| #41 | Phase 1 Exit | C3 | all five gates + P1.6-1/2 verified here |

### P2 — vision

| Fx | Ticket | Comp | Where |
|----|--------|------|-------|
| #42 | P2-1 SmolVLM fork | C1 | pull SmolVLM-256M from HF, fork head, train on cloud |
| #43 | P2-2 score primitive on images | C1→C2 | train on cloud; parity with text path |
| #44 | P2-3 visual data mix | C2 | manifest ≥300/category committed in repo |
| #45 | P2-4 calibration on vision | C1→C2 | extend C2 grid to vision; run on cloud |
| #46 | P2-5 modality axis router | C2→C3 | router code in repo; integration test here |
| #47 | P2-6 full grid on vision | C1→C2 | run full grid on cloud; rows in frozen format |
| #48 | P2-GATE | C3 | gate |

### P3 — scale

| Fx | Ticket | Comp | Where |
|----|--------|------|-------|
| #49 | P3-1 learned Router policy | C1→C3 | train policy on telemetry; shadow-traffic here |
| #50 | P3-2 quantization/exports | C3 | ONNX/CoreML/MLX exports + validation suite here on returned model |
| #51 | P3-3 telemetry loop | C3→C1 | scheduled pipeline here; adjudicated outcomes become training examples |
| #52 | P3-4 new-domain flywheel | C2 | repeatable checklist lives in repo; drives all three |

## What the agent must never do again

- no fake models / fake adapters / offline smoke gates masquerading as evals
- no running metrics on this laptop as a stand-in for the real grid — this
  device validates numbers produced by cloud-trained models
- no generating gate numbers that a cloud run didn't produce
- every report row must round-trip: trained in the notebook, measured by the
  imported C2 harness, handed back here, schema-validated, then gated
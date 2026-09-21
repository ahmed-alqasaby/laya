## Agent skills

### Issue tracker

Issues and specs live as GitHub issues, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles, defaults kept; the label string equals the role name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Workflow: three components

Read `docs/plan.md` before any ticket work — it maps every issue to one of:

1. **C1 Cloud trainer (Kaggle notebook)**: pulls models from Hugging Face and trains them.
2. **C2 Eval environment (this repo)**: grid, buckets, splits, datasets, metrics, schema — imported into the notebook.
3. **C3 Local device (this repo)**: model + files come back here for validation, inference, metrics, and gating.

No fake models, no offline smoke gates. Gate numbers only from cloud-trained models run through C2.
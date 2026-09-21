# evaldata/ — private eval sets for the frozen grid

Cells in `configs/eval_grid.yaml` with `source: local` read one JSONL file per
cell. Drop the human's private sets here (6 target-domain sets, the strict
novelty split, the ordinal/rubric sets, script-detection stress sets) and the
next cloud run picks them up automatically — no code change. Files here are
gitignored (*.jsonl).

## One instance per line

A line is a single **question instance** against a state. The `state` and
`question` fields are passed verbatim to `predict(state, {qname: question})`.

```json
{
  "id": "dom1-0007",
  "state": {"text": "Weekly report is late and the CFO is asking for it."},
  "qname": "priority",
  "qtype": "score",
  "question": {
    "type": "score",
    "instructions": "How urgent is this report?",
    "criteria": ["can wait", "this week", "today"]
  },
  "options": ["can wait", "this week", "today"],
  "option_levels": [1.0, 2.0, 3.0],
  "answer": "today",
  "answer_idx": 2,
  "ordinal_value": 3.0,
  "teacher": [0.1, 0.2, 0.7],
  "should_escalate": false,
  "novelty": false,
  "domain": "automation",
  "cardinality": 3
}
```

## Field semantics

| field | required | meaning |
|---|---|---|
| `id` | yes | stable unique id |
| `state` | yes | dict (or JSON string) passed as the model state |
| `qname` | yes | key used in the questions dict |
| `qtype` | yes | `choice` \| `noul` \| `score` (`noul` = yes/no single probability) |
| `question` | yes | the typed question object (`type`/`instructions`/`criteria`) |
| `options` | choice/noul | ordered option labels (order matters) |
| `option_levels` | score | numeric levels; default `[0,1,...,N-1]` for score with `options` |
| `answer` | yes | gold answer (label for choice/noul, level label for score) |
| `answer_idx` | choice/noul | gold index into `options` |
| `ordinal_value` | score | gold numeric value (used for ordinal MAE) |
| `teacher` | no | distribution over `options` (required for soft-acc / soft-dist metrics) |
| `should_escalate` | no | ground-truth escalate label (required for escalation-AUROC) |
| `novelty` | no | mark rows of a held-out novelty split |
| `domain` | yes | domain label |
| `cardinality` | no | option count (auto-derived if absent) |
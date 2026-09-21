from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "1.0"

METRIC_KEYS = (
    "argmax_acc",
    "soft_acc",
    "soft_acc_gold",
    "raw_ece",
    "postfit_ece",
    "temperature",
    "ordinal_mae",
    "escalation_auroc",
    "selective_auroc",
    "precision_at_escalation",
    "risk_coverage_auc",
    "option_order_robustness",
)

QUESTION_TYPES = ("choice", "score", "noul")


@dataclass
class ReportRow:
    schema_version: str = SCHEMA_VERSION
    grid_tag: str = ""
    cell_id: str = ""
    checkpoint: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict)
    n_instances: int = 0
    question_types: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    per_question_type: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    memory_mb: Optional[float] = None
    cost_per_1m_tokens: Optional[float] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema mismatch: {self.schema_version} != {SCHEMA_VERSION}")
        if not self.cell_id or not self.checkpoint:
            raise ValueError("row missing cell_id/checkpoint")
        for k in METRIC_KEYS:
            if k not in self.metrics:
                self.metrics[k] = None
        for k, v in self.metrics.items():
            if v is not None and not isinstance(v, (int, float)):
                raise ValueError(f"metric {k} must be a number or null, got {type(v)}")
        if self.metrics.get("temperature") is not None and self.metrics.get("postfit_ece") is None:
            raise ValueError("temperature present but postfit_ece missing")
        for qt in self.per_question_type:
            if qt not in QUESTION_TYPES:
                raise ValueError(f"unknown question type {qt}")


def rows_to_jsonl(rows: list[ReportRow], path: str) -> None:
    with open(path, "w") as f:
        for r in rows:
            r.validate()
            f.write(json.dumps(r.to_dict()) + "\n")


def rows_from_jsonl(path: str) -> list[ReportRow]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = ReportRow(**json.loads(line))
            out.append(r)
    return out


def render_metric_table(rows: list[ReportRow]) -> str:
    header = ["cell", "checkpoint", "n", "argmax_acc", "soft_acc", "soft_acc_gold", "raw_ece", "postfit_ece", "ordinal_mae", "sel_auroc"]
    lines = [" | ".join(header), " | ".join(["---"] * len(header))]
    for r in rows:
        m = r.metrics
        cells = [
            r.cell_id,
            r.checkpoint.get("variant", "?"),
            str(r.n_instances),
            _fmt(m.get("argmax_acc")),
            _fmt(m.get("soft_acc")),
            _fmt(m.get("soft_acc_gold")),
            _fmt(m.get("raw_ece")),
            _fmt(m.get("postfit_ece")),
            _fmt(m.get("ordinal_mae")),
            _fmt(m.get("selective_auroc")),
        ]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:.4f}"
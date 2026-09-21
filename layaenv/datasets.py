from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Inst:
    id: str
    state: Any
    qname: str
    qtype: str
    question: dict
    options: Optional[list[str]] = None
    option_levels: Optional[list[float]] = None
    answer: Optional[str] = None
    answer_idx: Optional[int] = None
    ordinal_value: Optional[float] = None
    teacher: Optional[list[float]] = None
    should_escalate: Optional[bool] = None
    domain: str = ""
    cardinality: int = 0
    novelty: bool = False
    meta: dict = field(default_factory=dict)

    @staticmethod
    def from_row(row: dict) -> "Inst":
        return Inst(**row)


def _as_state(state: Any) -> Any:
    if isinstance(state, str):
        try:
            return json.loads(state)
        except Exception:
            return {"text": state}
    return state


def _choice_color():
    return ""


def _question_options(question: dict) -> list[str]:
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        return list(criteria.keys())
    if isinstance(criteria, list):
        return [str(c) for c in criteria]
    return []


def typed_decisions_rows(ds_rows: list[dict]) -> list[Inst]:
    out: list[Inst] = []
    for r in ds_rows:
        state = _as_state(r.get("state"))
        questions = json.loads(r.get("questions") or "{}")
        gold = r.get("gold")
        if isinstance(gold, str):
            try:
                gold = json.loads(gold)
            except Exception:
                gold = {}
        domain = str(r.get("workflow") or "typed-decisions")
        for qname, q in questions.items():
            if not isinstance(q, dict):
                continue
            qtype = str(q.get("type") or "choice")
            options = _question_options(q) if qtype in ("choice", "noul") else None
            inst = Inst(
                id=f"{r.get('id')}__{qname}",
                state=state,
                qname=qname,
                qtype=qtype,
                question=q,
                options=options,
                domain=domain,
                cardinality=len(options) if options else 0,
            )
            if qtype in ("choice", "noul") and options is not None:
                gold_ans = None
                if isinstance(gold, dict):
                    gold_ans = gold.get(qname)
                if gold_ans is None:
                    gold_ans = r.get(f"{qname}__label")
                if gold_ans is not None and str(gold_ans) in options:
                    inst.answer = str(gold_ans)
                    inst.answer_idx = options.index(str(gold_ans))
                probs = r.get(f"{qname}__probabilities")
                if probs is not None:
                    if isinstance(probs, str):
                        try:
                            probs = json.loads(probs)
                        except Exception:
                            probs = None
                    if isinstance(probs, (list, tuple)) and len(probs) == len(options):
                        inst.teacher = [float(x) for x in probs]
            if qtype == "score":
                score = r.get(f"{qname}__score")
                if score is not None:
                    inst.ordinal_value = float(score)
                inst.option_levels = None
            out.append(inst)
    return out


def banking77_rows(ds_rows: list[dict], label_names: list[str]) -> list[Inst]:
    out: list[Inst] = []
    for r in ds_rows:
        text = str(r.get("text") or "")
        label = int(r.get("label") or 0)
        name = label_names[label] if 0 <= label < len(label_names) else f"intent_{label}"
        question = {
            "type": "choice",
            "instructions": "Which banking intent is this user request about?",
            "criteria": {n: n.replace("_", " ") for n in label_names},
        }
        options = _question_options(question)
        inst = Inst(
            id=f"banking77__{len(out)}",
            state={"text": text},
            qname="intent",
            qtype="choice",
            question=question,
            options=options,
            answer=name,
            answer_idx=label,
            domain="banking",
            cardinality=len(options),
        )
        out.append(inst)
    return out


def xnli_rows(ds_rows: list[dict]) -> list[Inst]:
    out: list[Inst] = []
    names = ["entailment", "neutral", "contradiction"]
    for r in ds_rows:
        premise = str(r.get("premise") or "")
        hypothesis = str(r.get("hypothesis") or "")
        label = int(r.get("label") or 0)
        question = {
            "type": "choice",
            "instructions": "Given the premise, how does the hypothesis relate to it?",
            "criteria": {n: n for n in names},
        }
        inst = Inst(
            id=f"xnli__{len(out)}",
            state={"premise": premise, "hypothesis": hypothesis},
            qname="relation",
            qtype="choice",
            question=question,
            options=names,
            answer=names[label],
            answer_idx=label,
            domain="nli",
            cardinality=3,
        )
        out.append(inst)
    return out


def sst5_rows(ds_rows: list[dict]) -> list[Inst]:
    out: list[Inst] = []
    levels = [0.0, 1.0, 2.0, 3.0, 4.0]
    for r in ds_rows:
        text = str(r.get("text") or "")
        label = int(r.get("label") or 0)
        question = {
            "type": "score",
            "instructions": "Rate the sentiment of the text from 0 (very negative) to 4 (very positive).",
            "criteria": [str(i) for i in levels],
        }
        inst = Inst(
            id=f"sst5__{len(out)}",
            state={"text": text},
            qname="sentiment",
            qtype="score",
            question=question,
            options=[str(int(l)) for l in levels],
            option_levels=levels,
            ordinal_value=float(label),
            answer=str(label),
            answer_idx=label,
            domain="sentiment",
            cardinality=5,
        )
        out.append(inst)
    return out


def local_jsonl_rows(path: str) -> list[Inst]:
    out: list[Inst] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            inst = Inst.from_row(row)
            inst.state = _as_state(inst.state)
            inst.question = inst.question or {}
            inst.options = inst.options or _question_options(inst.question)
            inst.domain = inst.domain or "local"
            inst.cardinality = inst.cardinality or (len(inst.options) if inst.options else 0)
            out.append(inst)
    return out
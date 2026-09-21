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


def _probs_by_options(probs: Any, options: Optional[list[str]]) -> Optional[list[float]]:
    """Teacher distribution from a {option_label -> probability} dict, aligned to options order."""
    if not isinstance(probs, dict) or not options:
        return None
    vals = [probs.get(o) for o in options]
    if any(not isinstance(v, (int, float)) for v in vals):
        return None
    return [float(v) for v in vals]


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
            g = gold.get(qname) if isinstance(gold, dict) else None
            if not isinstance(g, dict):
                g = {k: r.get(f"{qname}__{k}") for k in ("label", "score", "probabilities")}
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
                inst.answer = g.get("label")
                if inst.answer is not None and str(inst.answer) in options:
                    inst.answer_idx = options.index(str(inst.answer))
                inst.teacher = _probs_by_options(g.get("probabilities"), options)
            if qtype == "score":
                inst.answer = g.get("label")
                if isinstance(g.get("score"), (int, float)):
                    inst.ordinal_value = float(g["score"])
                crit = q.get("criteria")
                if isinstance(crit, list):
                    inst.option_levels = [float(i) for i in range(len(crit))]
                inst.teacher = _probs_by_options(g.get("probabilities"), [str(i) for i in range(len(crit))] if isinstance(crit, list) else None)
            out.append(inst)
    return out


BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support",
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
    "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card",
    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card",
    "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money",
    "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def banking77_rows(ds_rows: list[dict], label_names: list[str]) -> list[Inst]:
    out: list[Inst] = []
    label_names = label_names or BANKING77_LABELS
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
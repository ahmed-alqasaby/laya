from __future__ import annotations

import json
import os
import time
import zipfile
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from . import backends, datasets, metrics
from .grid import Cell, load_grid
from .schema import ReportRow, render_metric_table, rows_to_jsonl

BREAKDOWN_KEYS = ("argmax_acc", "soft_acc", "raw_ece", "postfit_ece", "ordinal_mae", "n")


def _normalize_instance(inst: datasets.Inst, out: dict) -> dict:
    qtype = inst.qtype
    norm = {
        "qtype": qtype,
        "probs": None,
        "conf": None,
        "pred_idx": None,
        "y_idx": None,
        "correct": None,
        "teacher": inst.teacher,
        "ord_expected": None,
        "ord_gold": inst.ordinal_value,
        "ord_levels": inst.option_levels,
        "boom": False,
    }
    if qtype == "choice":
        if out.get("probs") is not None and inst.options is not None:
            p = out["probs"]
            if len(p) == len(inst.options):
                norm["probs"] = p
                norm["conf"] = float(p.max())
                norm["pred_idx"] = int(p.argmax())
        if inst.answer is not None and inst.options is not None:
            try:
                norm["y_idx"] = inst.options.index(str(inst.answer))
            except ValueError:
                norm["y_idx"] = inst.answer_idx
            if norm["y_idx"] is None and isinstance(inst.answer_idx, int):
                norm["y_idx"] = inst.answer_idx
            if norm["pred_idx"] is not None and norm["y_idx"] is not None:
                norm["correct"] = bool(norm["pred_idx"] == norm["y_idx"])
    elif qtype == "noul":
        p = None
        if isinstance(out.get("noul"), (int, float)):
            p = float(out["noul"])
        elif out.get("probs") is not None and len(out["probs"]) == 2:
            p = float(out["probs"][1])
        if p is not None:
            norm["probs"] = np.array([1.0 - p, p])
            norm["conf"] = max(p, 1.0 - p)
            norm["pred_idx"] = 1 if p >= 0.5 else 0
        if inst.answer is not None:
            ans = str(inst.answer).lower()
            if ans in ("true", "1", "yes"):
                norm["y_idx"] = 1
            elif ans in ("false", "0", "no"):
                norm["y_idx"] = 0
            elif isinstance(inst.answer_idx, int):
                norm["y_idx"] = inst.answer_idx
            if norm["pred_idx"] is not None and norm["y_idx"] is not None:
                norm["correct"] = bool(norm["pred_idx"] == norm["y_idx"])
    elif qtype == "score":
        expected = None
        if isinstance(out.get("score"), (int, float)):
            expected = float(out["score"])
        if out.get("probs") is not None and inst.option_levels is not None:
            p = out["probs"]
            levels = np.asarray(inst.option_levels, dtype=float)
            if len(p) == len(levels):
                norm["probs"] = p
                norm["conf"] = float(p.max())
                if expected is None:
                    expected = float(np.dot(p, levels))
        norm["ord_expected"] = expected
        if inst.option_levels is not None:
            levels = np.asarray(inst.option_levels, dtype=float)
            if norm["probs"] is not None:
                norm["pred_idx"] = int(np.argmax(norm["probs"]))
            elif expected is not None:
                norm["pred_idx"] = int(np.abs(levels - expected).argmin())
            if inst.ordinal_value is not None:
                norm["y_idx"] = int(np.abs(levels - inst.ordinal_value).argmin())
            elif isinstance(inst.answer_idx, int):
                norm["y_idx"] = inst.answer_idx
            if norm["pred_idx"] is not None and norm["y_idx"] is not None:
                norm["correct"] = bool(norm["pred_idx"] == norm["y_idx"])
    return norm


def load_cell_instances(cell: Cell, repo_root: str) -> tuple[list[datasets.Inst], list[str]]:
    if cell.source == "local":
        path = os.path.join(repo_root, cell.local_file)
        rows = datasets.local_jsonl_rows(path)
        if not rows:
            return [], [f"local file {cell.local_file} missing or empty (skipped)"]
        if cell.max_instances:
            rows = rows[: cell.max_instances]
        return rows, []
    import datasets as hf_datasets

    try:
        if cell.hf_config:
            hf = hf_datasets.load_dataset(cell.hf_repo, cell.hf_config, split=cell.hf_split)
        else:
            hf = hf_datasets.load_dataset(cell.hf_repo, split=cell.hf_split)
    except Exception as exc:  # noqa: BLE001
        return [], [f"failed to load {cell.hf_repo}: {exc}"]
    rows = list(hf)
    if cell.max_instances:
        rows = rows[: cell.max_instances]
    proc = cell.processor
    if proc == "typed_decisions":
        insts = datasets.typed_decisions_rows(rows)
    elif proc == "banking77":
        names = getattr(hf, "features", {}).get("label", None)
        label_names = list(names.names) if names is not None and hasattr(names, "names") else []
        insts = datasets.banking77_rows(rows, label_names)
    elif proc == "xnli":
        insts = datasets.xnli_rows(rows)
    elif proc == "sst5":
        insts = datasets.sst5_rows(rows)
    else:
        return [], [f"unknown processor {cell.processor} (skipped)"]
    if cell.max_instances:
        insts = insts[: cell.max_instances]
    return insts, []


def _run_predict(backend: backends.Backend, inst: datasets.Inst) -> tuple[Optional[float], Optional[dict]]:
    questions = {inst.qname: inst.question}
    t0 = time.perf_counter()
    try:
        res = backend.predict(inst.state, questions)
    except Exception as exc:  # noqa: BLE001
        return 0.0, {"error": str(exc)}
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, backends.extract_answer(res, inst.qname)


def _group_key(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, default=str)


def _group_instances(insts: list[datasets.Inst]) -> list[list[tuple[int, datasets.Inst]]]:
    groups: dict[str, list[tuple[int, datasets.Inst]]] = {}
    order: list[str] = []
    for idx, inst in enumerate(insts):
        key = _group_key(inst.state)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((idx, inst))
    return [groups[k] for k in order]


def _overall_metrics(ok: list[dict]) -> dict:
    m: dict[str, Any] = {k: None for k in (
        "argmax_acc", "soft_acc", "raw_ece", "postfit_ece", "temperature", "ordinal_mae",
        "escalation_auroc", "selective_auroc", "precision_at_escalation", "risk_coverage_auc",
        "option_order_robustness")}

    scored = [n for n in ok if n["qtype"] == "score" and n["ord_expected"] is not None and n["ord_gold"] is not None]
    if scored:
        m["ordinal_mae"] = float(np.mean([abs(n["ord_expected"] - n["ord_gold"]) for n in scored]))

    mc = [n for n in ok if n["probs"] is not None and n["y_idx"] is not None and n["correct"] is not None]
    if mc:
        probs = [np.asarray(n["probs"], dtype=float) for n in mc]
        y = [int(n["y_idx"]) for n in mc]
        m["argmax_acc"] = float(np.mean([int(p.argmax()) == yv for p, yv in zip(probs, y)]))
        m["raw_ece"] = metrics.rows_cal_ece(probs, y)
        m["temperature"], m["postfit_ece"] = metrics.rows_fit_temperature(probs, y)
        with_t = [n for n in mc if n["teacher"] is not None]
        if with_t:
            m["soft_acc"] = metrics.rows_soft_acc(
                [np.asarray(n["probs"], dtype=float) for n in with_t],
                [np.asarray(n["teacher"], dtype=float) for n in with_t],
            )

    pairs = [(n["conf"], n["correct"]) for n in ok if n["conf"] is not None and n["correct"] is not None]
    if pairs:
        conf = np.array([p[0] for p in pairs])
        correct = np.array([bool(p[1]) for p in pairs])
        m["selective_auroc"] = metrics.auroc(conf, ~correct)
        m["risk_coverage_auc"] = metrics.risk_coverage_auc(conf, correct)
        esc = np.array([bool(n["escalate"]) if n.get("escalate") is not None else False for n in ok])
        if esc.any() and (esc.sum() != len(esc)):
            if len(esc) == len(ok):
                auroc_esc = metrics.auroc(conf, esc[: len(conf)])
                m["escalation_auroc"] = auroc_esc
            m["precision_at_escalation"] = None
    return m


def _per_question_type(ok: list[dict]) -> dict:
    out = {}
    for qt in ("choice", "score", "noul"):
        sel = [n for n in ok if n["qtype"] == qt]
        d: dict[str, Any] = {"argmax_acc": None, "soft_acc": None, "raw_ece": None, "postfit_ece": None, "ordinal_mae": None, "n": len(sel)}
        if not sel:
            out[qt] = d
            continue
        accs = [n["correct"] for n in sel if n["correct"] is not None]
        if accs:
            d["argmax_acc"] = float(np.mean(accs))
        mc = [n for n in sel if n["probs"] is not None and n["y_idx"] is not None and n["correct"] is not None]
        if mc:
            probs = [np.asarray(n["probs"], dtype=float) for n in mc]
            y = [int(n["y_idx"]) for n in mc]
            d["raw_ece"] = metrics.rows_cal_ece(probs, y)
            t, post = metrics.rows_fit_temperature(probs, y)
            if post is not None:
                d["postfit_ece"] = post
            with_t = [n for n in mc if n["teacher"] is not None]
            if with_t:
                d["soft_acc"] = metrics.rows_soft_acc(
                    [np.asarray(n["probs"], dtype=float) for n in with_t],
                    [np.asarray(n["teacher"], dtype=float) for n in with_t],
                )
        if qt == "score":
            scored = [n for n in sel if n["ord_expected"] is not None and n["ord_gold"] is not None]
            if scored:
                d["ordinal_mae"] = float(np.mean([abs(n["ord_expected"] - n["ord_gold"]) for n in scored]))
        out[qt] = d
    return out


def _perm_robustness(backend: backends.Backend, insts: list[datasets.Inst], ok: list[dict], rng, k: int = 200) -> Optional[float]:
    indices = [i for i, n in enumerate(ok) if insts[i].qtype != "score" and insts[i].options and len(insts[i].options) >= 2 and ok[i].get("pred_idx") is not None]
    if not indices:
        return None
    sample = rng.choice(indices, size=min(k, len(indices)), replace=False)
    agrees: list[bool] = []
    for i in sample:
        inst = insts[int(i)]
        perm = list(inst.options)
        rng.shuffle(perm)
        try:
            _, out2 = _run_predict(backend, _perm_inst(inst, perm))
        except Exception:  # noqa: BLE001
            continue
        n2 = _normalize_instance(_perm_inst(inst, perm), out2)
        orig_label = inst.options[ok[int(i)]["pred_idx"]]
        if n2.get("pred_idx") is None:
            continue
        perm_label = perm[n2["pred_idx"]]
        agrees.append(orig_label == perm_label)
    return float(np.mean(agrees)) if agrees else None


def _perm_inst(inst: datasets.Inst, perm: list[str]) -> datasets.Inst:
    import copy

    q = copy.deepcopy(inst.question)
    q["criteria"] = {orig: perm[j] for j, orig in enumerate(inst.options)}
    new = copy.copy(inst)
    new.question = q
    new.options = perm
    new.teacher = None
    return new


def run_cell(cell: Cell, backend: backends.Backend, insts: list[datasets.Inst], run: dict, perm_sample: int = 200, seed: int = 0) -> ReportRow:
    rng = np.random.default_rng(seed)
    ok: list[dict] = []
    times_ms: list[float] = []
    for group in _group_instances(insts):
        state = group[0][1].state
        questions = {inst.qname: inst.question for _, inst in group}
        t0 = time.perf_counter()
        try:
            res = backend.predict(state, questions)
        except Exception as exc:  # noqa: BLE001
            dt = 0.0
            for idx, inst in group:
                ok.append({"qtype": inst.qtype, "error": str(exc)})
        else:
            dt = (time.perf_counter() - t0) * 1000.0
            for idx, inst in group:
                out = backends.extract_answer(res, inst.qname)
                if "error" in out:
                    ok.append({"qtype": inst.qtype, "error": True})
                else:
                    ok.append(_normalize_instance(inst, out))
        times_ms.append(dt)

    notes = []
    if len(ok) < len(insts):
        notes.append(f"{len(insts) - len(ok)} predict calls failed")
    if backend.subfolder:
        notes.append(f"model repo {backend.repo}/{backend.subfolder}")
    else:
        notes.append(f"model repo {backend.repo}")

    m = _overall_metrics(ok)
    per_qt = _per_question_type(ok)
    if perm_sample:
        m["option_order_robustness"] = _perm_robustness(backend, insts, ok, rng, perm_sample)

    model_times = np.array([t for t in times_ms if t > 0])
    latency = metrics.batch_stats(model_times) if len(model_times) else {}

    row = ReportRow(
        grid_tag=run.get("grid_tag", ""),
        cell_id=cell.id,
        checkpoint={"repo": backend.repo, "subfolder": backend.subfolder, "variant": backend.variant},
        run=run,
        n_instances=len(ok),
        question_types=cell.question_types,
        metrics=m,
        per_question_type=per_qt,
        latency_ms=latency,
        notes=notes,
    )
    return row


def run_grid(
    grid_path: str,
    repo_root: str,
    outdir: str,
    model_registry: dict,
    device: Optional[str] = None,
    variants: Optional[list[str]] = None,
    force_variants: Optional[list[str]] = None,
    perm_sample: int = 200,
    seed: int = 0,
) -> tuple[list[ReportRow], list[str]]:
    grid_tag, cells = load_grid(grid_path)
    os.makedirs(outdir, exist_ok=True)
    run = {"grid_tag": grid_tag, "date": datetime.now(timezone.utc).isoformat(), "commit": _git_commit(repo_root)}
    rows: list[ReportRow] = []
    skipped: list[str] = []
    for cell in cells:
        if force_variants:
            cell_models = list(force_variants)
        elif variants:
            cell_models = [m for m in cell.models if m in variants]
        else:
            cell_models = cell.models
        if not cell_models:
            skipped.append(f"cell {cell.id}: no requested model variants")
            continue
        insts, skip = load_cell_instances(cell, repo_root)
        if skip:
            skipped.extend(skip)
            continue
        for variant in cell_models:
            cfg = model_registry.get(variant)
            if cfg is None:
                skipped.append(f"cell {cell.id}: unknown variant {variant}")
                continue
            repo, subfolder = cfg
            backend = backends.load_backend(variant, repo, subfolder, device=device)
            row = run_cell(cell, backend, insts, run, perm_sample=perm_sample, seed=seed)
            rows.append(row)
    rows_to_jsonl(rows, os.path.join(outdir, "report.jsonl"))
    return rows, skipped


def _git_commit(repo_root: str) -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "-C", repo_root, "rev-parse", "HEAD"], text=True).strip()[:12]
    except Exception:  # noqa: BLE001
        return "unknown"


def write_artifacts(outdir: str, rows: list[ReportRow], skipped: list[str], extra: dict | None = None) -> str:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("# Layenv eval summary\n\n" + render_metric_table(rows) + "\n")
    meta = {"rows": len(rows), "skipped_cells": skipped}
    if extra:
        meta.update(extra)
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    zip_path = os.path.join(outdir, "laya-artifacts.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("report.jsonl", "summary.md", "run_meta.json"):
            p = os.path.join(outdir, name)
            if os.path.exists(p):
                z.write(p, arcname=name)
        for extra_name in ("probe.jsonl", "sample_predictions.jsonl", "telemetry.jsonl"):
            p = os.path.join(outdir, extra_name)
            if os.path.exists(p):
                z.write(p, arcname=extra_name)
    return zip_path
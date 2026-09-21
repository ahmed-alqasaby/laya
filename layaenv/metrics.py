from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata


def _check(probs: np.ndarray, shapes: tuple) -> bool:
    return probs is not None and len(probs) > 0 and probs.ndim == 1 and probs.shape == shapes


def cal_ece(probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 15) -> float | None:
    probs = np.asarray(probs, dtype=float)
    y_idx = np.asarray(y_idx)
    if len(probs) == 0 or probs.ndim != 2:
        return None
    conf = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    acc = np.zeros(n_bins)
    avg_conf = np.zeros(n_bins)
    weight = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            sel = conf <= hi
        else:
            sel = (conf > lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        weight[i] = sel.mean()
        acc[i] = correct[sel].mean()
        avg_conf[i] = conf[sel].mean()
    w = weight.sum()
    if w == 0:
        return None
    return float(np.sum(weight * np.abs(acc - avg_conf)))


def rows_cal_ece(probs: list[np.ndarray], y_idx: list | np.ndarray, n_bins: int = 15) -> float | None:
    """Calibration across a *list* of varying-width distributions (different option counts)."""
    if not probs:
        return None
    conf = np.array([p.max() for p in probs], dtype=float)
    correct = np.array([int(p.argmax()) == int(yv) for p, yv in zip(probs, y_idx)], dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    acc = np.zeros(n_bins)
    avg_conf = np.zeros(n_bins)
    weight = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = conf <= hi if i == 0 else (conf > lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        weight[i] = sel.mean()
        acc[i] = correct[sel].mean()
        avg_conf[i] = conf[sel].mean()
    w = weight.sum()
    if w == 0:
        return None
    return float(np.sum(weight * np.abs(acc - avg_conf)))


def rows_fit_temperature(probs: list[np.ndarray], y_idx: list | np.ndarray) -> tuple[float | None, float | None]:
    """Post-fit temperature across varying-width distributions, n-weighted across option-count groups."""
    by_k: dict[int, list[int]] = {}
    for i, p in enumerate(probs):
        by_k.setdefault(len(p), []).append(i)
    n = len(probs)
    temps, post_eces, wts = [], [], []
    for k, idx in by_k.items():
        if len(idx) < 2:
            continue
        P = np.array([probs[i] for i in idx], dtype=float)
        Y = np.array([y_idx[i] for i in idx])
        logits = probs_to_logits(P)
        t = fit_temperature(logits, Y)
        if t is None:
            continue
        e = cal_ece(scaled_probs(logits, t), Y)
        temps.append(t)
        wts.append(len(idx))
        post_eces.append(e if e is not None else float("nan"))
    if not temps:
        return None, None
    w = np.sum(wts)
    t_out = float(np.average(temps, weights=wts))
    e_out = float(np.average([e for e in post_eces if not np.isnan(e)], weights=[w for w, e in zip(wts, post_eces) if not np.isnan(e)])) if any(not np.isnan(e) for e in post_eces) else None
    return (t_out, e_out) if e_out is not None else (t_out, None)


def rows_soft_acc(preds: list[np.ndarray], teachers: list[np.ndarray]) -> float | None:
    vals: list[float] = []
    for p, q in zip(preds, teachers):
        if p.shape != q.shape:
            continue
        vals.append(1.0 - 0.5 * float(np.abs(p - q).sum()))
    return float(np.mean(vals)) if vals else None


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float | None:
    logits = np.asarray(logits, dtype=float)
    y_idx = np.asarray(y_idx)
    if len(logits) == 0 or logits.ndim != 2:
        return None

    def nll(t: float) -> float:
        scaled = logits / t
        lg = logsumexp(scaled, axis=1)
        return float(-np.mean(scaled[np.arange(len(y_idx)), y_idx] - lg))

    res = minimize(nll, x0=np.array([1.0]), bounds=[(1e-4, 1e3)], method="L-BFGS-B")
    if not res.success:
        return 1.0
    return float(res.x[0])


def scaled_probs(logits: np.ndarray, t: float) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=float) / t, axis=1)


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)


def soft_acc(pred: np.ndarray, teacher: np.ndarray) -> float | None:
    pred = np.asarray(pred, dtype=float)
    teacher = np.asarray(teacher, dtype=float)
    if pred.ndim != 2 or pred.shape != teacher.shape or len(pred) == 0:
        return None
    tv = 0.5 * np.abs(pred - teacher).sum(axis=1)
    return float(np.mean(1.0 - tv))


def ordinal_mae(probs: np.ndarray, levels: np.ndarray, gold: np.ndarray) -> float | None:
    probs = np.asarray(probs, dtype=float)
    gold = np.asarray(gold, dtype=float)
    if probs.ndim != 2 or len(probs) == 0 or levels.size != probs.shape[1]:
        return None
    expected = probs @ levels
    return float(np.mean(np.abs(expected - gold)))


def auroc(score: np.ndarray, label: np.ndarray) -> float | None:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)
    pos = label == 1
    neg = label == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(np.r_[score[pos], score[neg]])
    u = float(ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2)
    return u / (n_pos * n_neg)


def risk_coverage_auc(conf: np.ndarray, correct: np.ndarray) -> float | None:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(conf) == 0:
        return None
    order = np.argsort(-conf)
    n = len(conf)
    cov = np.arange(1, n + 1) / n
    err = np.cumsum(~correct[order]) / np.arange(1, n + 1)
    return float(np.trapezoid(err, cov))


def precision_at_escalation(
    score: np.ndarray, escalate: np.ndarray, quantile: float = 0.05
) -> tuple[float, float, float] | None:
    score = np.asarray(score, dtype=float)
    escalate = np.asarray(escalate, dtype=bool)
    if len(score) == 0 or escalate.sum() == 0 or escalate.sum() == len(escalate):
        return None
    tau = np.quantile(score, 1.0 - quantile)
    esc = score >= tau
    if esc.sum() == 0:
        return None
    precision = float((esc & escalate).sum() / esc.sum())
    return precision, float(esc.mean()), float(tau)


def perm_robustness(agree: np.ndarray, tv: np.ndarray) -> float | None:
    agree = np.asarray(agree)
    tv = np.asarray(tv)
    if len(agree) == 0:
        return None
    return float(agree.mean()), float(tv.mean())


def batch_stats(times_ms: np.ndarray) -> dict:
    t = np.asarray(times_ms, dtype=float)
    if len(t) == 0:
        return {"mean_ms": None, "p50_ms": None, "calls_per_sec": None}
    return {
        "mean_ms": float(t.mean()),
        "p50_ms": float(np.median(t)),
        "calls_per_sec": float(1000.0 / t.mean()),
    }
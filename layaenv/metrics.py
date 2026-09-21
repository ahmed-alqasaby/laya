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
from __future__ import annotations

import os
import time
from typing import Any, Optional

os.environ.setdefault("USE_TF", "0")


class Backend:
    def __init__(self, agent: Any, variant: str, repo: str, subfolder: Optional[str] = None, device: Optional[str] = None):
        self.agent = agent
        self.variant = variant
        self.repo = repo
        self.subfolder = subfolder
        self.device = device

    def predict(self, state: Any, questions: dict) -> dict:
        return self.agent.predict(state, questions)


def load_agent(repo: str, subfolder: Optional[str] = None) -> Any:
    import laya

    if subfolder:
        try:
            return laya.load(repo, subfolder=subfolder)
        except TypeError:
            return laya.load(f"{repo}/{subfolder}")
    return laya.load(repo)


def load_backend(
    variant: str,
    repo: str,
    subfolder: Optional[str] = None,
    device: Optional[str] = None,
) -> Backend:
    import laya

    if variant == "router":
        router = laya.Router(preload=True)
        if device:
            try:
                router = laya.Router(preload=True, device=device)
            except TypeError:
                pass
        return Backend(router, "router", repo, None, device)
    agent = load_agent(repo, subfolder)
    return Backend(agent, variant, repo, subfolder, device)


def extract_answer(res: dict, qname: str) -> dict:
    out = {"choice": None, "score": None, "noul": None, "probs": None, "raw": None}
    try:
        ans = res["answers"][qname]
    except Exception:
        return out
    out["raw"] = ans
    if not isinstance(ans, dict):
        if isinstance(ans, (int, float)):
            out["score"] = float(ans)
        return out
    for k in ("choice", "score", "noul"):
        if k in ans:
            v = ans[k]
            out[k] = float(v) if isinstance(v, (int, float)) and k != "choice" else v
    for probe in ("probabilities", "distribution", "probs", "prob", "scores", "logits"):
        v = ans.get(probe)
        if v is None:
            continue
        if isinstance(v, str):
            import json

            try:
                v = json.loads(v)
            except Exception:
                continue
        try:
            import numpy as np

            if isinstance(v, dict):
                v = list(v.values())
            arr = np.asarray(v, dtype=float).ravel()
            if arr.size == 0:
                continue
            if arr.min() < 0 or not np.isclose(arr.sum(), 1.0, atol=0.15):
                e = np.exp(arr - arr.max())
                arr = e / e.sum()
            else:
                arr = arr / arr.sum()
            out["probs"] = arr
            break
        except Exception:
            continue
    return out


def probe_result(backend: Backend, state: Any, questions: dict) -> dict:
    t0 = time.perf_counter()
    try:
        res = backend.predict(state, questions)
        ok = True
    except Exception as exc:  # noqa: BLE001
        res = {"error": str(exc)}
        ok = False
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {"ok": ok, "elapsed_ms": dt_ms, "result": res}
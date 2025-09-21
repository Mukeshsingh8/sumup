# src/policy.py
import re, time
from typing import Dict, Any, List
import pandas as pd

from .rules import check_rules
from .features import featurize_one

PII_REDS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    (re.compile(r"\b\d{10,16}\b"), "<NUMBER>"),
]

def redact(s: str) -> str:
    t = s or ""
    for rx, repl in PII_REDS: t = rx.sub(repl, t)
    return t

def decide(event: Dict[str, Any],
           conv_state: Dict[str, Any],
           artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input event:
      {conversation_id, role, message, ts, lang, prev_bot_text}
    """
    t0 = time.time()
    model = artifacts["model"]
    feat_order = artifacts["feature_order"]
    tau = artifacts["tau"]
    policy = artifacts["policy"]
    guards = (policy.get("guards") or {})
    min_turn_before_model = int(guards.get("min_turn_before_model", 0))

    cid = event["conversation_id"]
    role = event["role"]
    user_text = event["message"] if role == "user" else ""
    prev_bot_text = event.get("prev_bot_text", "")
    user_turn_idx = int(conv_state.get("user_turn_idx", 0))

    fired = check_rules(user_text, prev_bot_text, policy)
    if "explicit_human_request" in fired or "risk_terms" in fired or "frustration_detected" in fired:
        where = "rules"
        decision = True
        score = 1.0
    else:
        if user_turn_idx < min_turn_before_model:
            where = "guard"
            decision = False
            score = 0.0
        else:
            row, conv_state = featurize_one(user_turn_idx, user_text, prev_bot_text, conv_state, policy, feat_order)
            p = float(model.predict_proba(row)[:,1][0])
            decision = (p >= tau)
            score = p
            where = "model"

    # update turn counter if user
    if role == "user":
        conv_state["user_turn_idx"] = user_turn_idx + 1

    latency_ms = int((time.time() - t0) * 1000)
    return {
        "conversation_id": cid,
        "turn_id": event.get("turn_id"),
        "escalate": bool(decision),
        "where": where,
        "score": float(score),
        "threshold": float(tau),
        "fired_rules": fired,
        "reason": ("user explicitly requested human" if "explicit_human_request" in fired else
                   "risk term present" if "risk_terms" in fired else
                   "guard: too early for model" if where == "guard" else
                   "model score >= tau" if decision else "model score < tau"),
        "latency_ms": latency_ms,
        "model_version": event.get("model_version", "model.joblib"),
        "policy_version": event.get("policy_version", "policy@assess"),
        "redacted_user_text": redact(user_text),
        "redacted_bot_text": redact(prev_bot_text),
        "state": {
            "user_turn_idx": int(conv_state.get("user_turn_idx", 0)),
            "no_progress_count": float(conv_state.get("no_progress_count", 0.0)),
            "bot_repeat_count": float(conv_state.get("bot_repeat_count", 0.0)),
        }
    }, conv_state

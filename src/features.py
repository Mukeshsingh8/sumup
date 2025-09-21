# src/features.py
import re
import pandas as pd
from typing import Dict, List, Any

def _has_any(patterns: List[str], s: str) -> int:
    s = (s or "").lower()
    return int(any(re.search(p, s) for p in patterns))

def _caps_ratio(s: str) -> float:
    if not s: return 0.0
    caps = sum(1 for c in s if c.isupper())
    letters = sum(1 for c in s if c.isalpha())
    return (caps / letters) if letters else 0.0

def featurize_one(user_turn_idx: int, user_text: str, prev_bot_text: str,
                  conv_state: Dict[str, Any], policy: Dict[str, Any],
                  feature_order: List[str]) -> (pd.DataFrame, Dict[str, Any]):
    rules = (policy.get("rules") or {})
    unhelp = (rules.get("bot_unhelpful_templates") or {}).get("patterns", [])
    ask_human = (rules.get("explicit_human_request") or {}).get("patterns", [])
    risk = (rules.get("risk_terms") or {}).get("patterns", [])

    X = {
        "turn_idx": float(user_turn_idx),
        "user_caps_ratio": float(_caps_ratio(user_text)),
        "exclam_count": float((user_text or "").count("!")),
        "msg_len": float(len(user_text or "")),
        "bot_unhelpful": float(_has_any(unhelp, prev_bot_text)),
        "user_requests_human": float(_has_any(ask_human, user_text)),
        "risk_terms": float(_has_any(risk, user_text)),
        "no_progress_count": float(conv_state.get("no_progress_count", 0.0)),
        "bot_repeat_count": float(conv_state.get("bot_repeat_count", 0.0)),
    }

    # update rolling state
    this_bot = (prev_bot_text or "").strip().lower()
    prev_bot = conv_state.get("prev_bot_text", "")
    if prev_bot and this_bot and (this_bot == prev_bot):
        conv_state["bot_repeat_count"] = conv_state.get("bot_repeat_count", 0.0) + 1.0
    else:
        conv_state["bot_repeat_count"] = max(conv_state.get("bot_repeat_count", 0.0) - 1.0, 0.0)
    if _has_any(unhelp, this_bot):
        conv_state["no_progress_count"] = conv_state.get("no_progress_count", 0.0) + 1.0
    else:
        conv_state["no_progress_count"] = max(conv_state.get("no_progress_count", 0.0) - 1.0, 0.0)
    conv_state["prev_bot_text"] = this_bot

    row = pd.DataFrame([[X[k] for k in feature_order]], columns=feature_order)
    return row, conv_state

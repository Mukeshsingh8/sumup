# src/rules.py
import re
from typing import List, Dict, Any

def _has_any(patterns: List[str], s: str) -> bool:
    s = (s or "").lower()
    return any(re.search(p, s) for p in patterns)

def check_rules(user_text: str, prev_bot_text: str, policy: Dict[str, Any]) -> List[str]:
    rules = (policy.get("rules") or {})
    fired = []
    if rules.get("explicit_human_request", {}).get("enabled", True):
        patt = rules["explicit_human_request"].get("patterns", [])
        if _has_any(patt, user_text): fired.append("explicit_human_request")
    if rules.get("risk_terms", {}).get("enabled", True):
        patt = rules["risk_terms"].get("patterns", [])
        if _has_any(patt, user_text): fired.append("risk_terms")
    if rules.get("bot_unhelpful_templates", {}).get("enabled", True):
        patt = rules["bot_unhelpful_templates"].get("patterns", [])
        if _has_any(patt, prev_bot_text): fired.append("bot_unhelpful_template_seen")
    return fired

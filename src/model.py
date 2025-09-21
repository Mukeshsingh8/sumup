# src/model.py
import os, json, joblib
import pandas as pd
from typing import Dict, Any, List, Tuple

def load_artifacts(art_dir: str) -> Tuple[object, List[str], float, Dict[str, Any]]:
    model = joblib.load(os.path.join(art_dir, "model.joblib"))
    with open(os.path.join(art_dir, "feature_order.json"), "r", encoding="utf-8") as f:
        feat_order = json.load(f)
    tau = 0.5
    with open(os.path.join(art_dir, "version.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("threshold="):
                tau = float(line.strip().split("=",1)[1]); break
    # prefer repo policy.yaml; else use snapshot
    policy_path = "policy.yaml"
    if not os.path.exists(policy_path):
        policy_path = os.path.join(art_dir, "policy.yaml")
    try:
        import yaml
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = yaml.safe_load(f) or {}
    except Exception:
        policy = {}
    return model, feat_order, tau, policy

def predict_proba(model, X: pd.DataFrame) -> float:
    return float(model.predict_proba(X)[:,1][0])

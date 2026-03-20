from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from pathlib import Path as _Path

ROOT_DIR = _Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config


def load_model(model_path: Path | None = None) -> Any:
    """Load the trained champion pipeline (preferred path with fallback)."""
    if model_path is None:
        model_path = config.MODEL_PATH

    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        with open(config.ROOT_MODEL_FALLBACK_PATH, "rb") as f:
            return pickle.load(f)


def predict_risk(inputs: dict[str, float], model: Any | None = None) -> dict[str, Any]:
    """
    Predict diabetes risk probability and map it to Low/Medium/High risk.

    Returns a dict:
      - prob_class_1: probability of Outcome=1
      - risk_pct: prob * 100
      - risk_level: Low / Medium / High
      - pred_label: Diabetes / Non-Diabetes
    """
    if model is None:
        model = load_model()

    input_df = pd.DataFrame([{k: inputs[k] for k in config.FEATURE_COLUMNS}])
    prob_class_1 = float(model.predict_proba(input_df)[0][1])
    risk_pct = prob_class_1 * 100.0

    pred_label = "Diabetes" if prob_class_1 >= config.DIABETIC_PROB_THRESHOLD else "No Diabetes"
    if risk_pct < config.RISK_LOW_MAX:
        risk_level = "Low"
    elif risk_pct < config.RISK_MODERATE_MAX:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "prob_class_1": prob_class_1,
        "risk_pct": risk_pct,
        "pred_label": pred_label,
        "risk_level": risk_level,
    }


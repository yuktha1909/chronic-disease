from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import config
from model.preprocessing import make_numeric_preprocess


def load_dataset() -> pd.DataFrame:
    """Load dataset from preferred path with URL fallback."""
    try:
        return pd.read_csv(config.DATASET_PATH)
    except FileNotFoundError:
        try:
            return pd.read_csv(config.ROOT_DATASET_FALLBACK_PATH)
        except FileNotFoundError:
            return pd.read_csv(config.DATASET_URL)

def train_and_select_champion(df: pd.DataFrame, test_size: float = 0.20):
    feature_cols = config.FEATURE_COLUMNS
    target_col = config.TARGET_COLUMN

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE
    )
    preprocess = make_numeric_preprocess(feature_cols)

    cfgs = {
        "Logistic Regression": {
            "est": LogisticRegression(max_iter=500, solver="lbfgs"),
            "grid": {"clf__C": [0.1, 1, 10], "clf__penalty": ["l2"]},
        },
        "Random Forest": {
            "est": RandomForestClassifier(random_state=config.RANDOM_STATE),
            "grid": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
            },
        },
        "SVM (RBF)": {
            "est": SVC(probability=True, random_state=config.RANDOM_STATE),
            # Keep the grid small; SVC(probability=True) can be expensive.
            "grid": {
                "clf__C": [0.5, 1.0, 2.0],
                "clf__gamma": ["scale", "auto"],
            },
        },
    }

    results: dict[str, dict] = {}
    for name, cfg in cfgs.items():
        pipe = Pipeline([("pre", preprocess), ("clf", cfg["est"])])
        gs = GridSearchCV(pipe, cfg["grid"], cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        y_pred = best.predict(X_test)

        results[name] = {
            "model": best,
            "y_pred": y_pred,
            "metrics": {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            },
            "best_params": gs.best_params_,
        }

    champion_name = max(results, key=lambda k: results[k]["metrics"]["accuracy"])
    best_model = results[champion_name]["model"]

    # Feature importance for the UI (permutation importance on the test set)
    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="accuracy",
        n_repeats=10,
        random_state=config.RANDOM_STATE,
    )
    feature_importance = {
        feature: float(perm.importances_mean[i])
        for i, feature in enumerate(feature_cols)
    }

    return results, champion_name, best_model, X_test, y_test, feature_importance


def train_and_save(model_path: Path | None = None, artifacts_dir: Path | None = None) -> Path:
    if model_path is None:
        model_path = config.MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    results, champion_name, champion_model, X_test, y_test, feature_importance = train_and_select_champion(df)

    if artifacts_dir is None:
        artifacts_dir = config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics + training summary (JSON)
    metrics_out: dict[str, dict] = {}
    for model_name, data in results.items():
        metrics_out[model_name] = {
            "metrics": data["metrics"],
            "best_params": data["best_params"],
        }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    with open(config.FEATURE_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)

    training_summary = {
        "best_model": champion_name,
        "selection_metric": config.MODEL_SELECTION_METRIC,
        "test_metrics": results[champion_name]["metrics"],
    }
    with open(config.TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    with open(model_path, "wb") as f:
        pickle.dump(champion_model, f)

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save champion model.")
    parser.add_argument("--retrain", action="store_true", help="Train and overwrite artifacts even if model exists.")
    args = parser.parse_args()

    if not args.retrain and config.MODEL_PATH.exists() and config.METRICS_PATH.exists():
        print("Model already exists. Use --retrain to retrain.")
    else:
        out = train_and_save()
        print(f"Saved champion model -> {out}")


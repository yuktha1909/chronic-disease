from pathlib import Path

# Project root (the folder containing this config.py)
ROOT_DIR = Path(__file__).resolve().parent

# Data / model locations (preferred)
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"

DATASET_PATH = DATA_DIR / "dataset.csv"
MODEL_PATH = MODEL_DIR / "model.pkl"

# Training artifacts (saved during `model/train.py`)
ARTIFACTS_DIR = MODEL_DIR / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"
TRAINING_SUMMARY_PATH = ARTIFACTS_DIR / "training_summary.json"

# Fallbacks (keep your existing files working without requiring moves)
ROOT_DATASET_FALLBACK_PATH = ROOT_DIR / "diabetes.csv"
ROOT_MODEL_FALLBACK_PATH = ROOT_DIR / "diabetes_model.pkl"

# Public dataset URL used when no local dataset is found
DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

# Reproducibility
RANDOM_STATE = 42

# Dataset schema
TARGET_COLUMN = "Outcome"
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "Age",
]

# Risk thresholds used by the UI (percent probability of class=1)
RISK_LOW_MAX = 35.0
RISK_MODERATE_MAX = 65.0

# Model selection: best model is chosen using accuracy on the held-out test split
MODEL_SELECTION_METRIC = "accuracy"

# Prediction threshold: prob(class=1) >= 0.5 => Diabetic
DIABETIC_PROB_THRESHOLD = 0.5

# Input validation ranges for the UI (used by utils.validation)
INPUT_RANGES = {
    "Pregnancies": {"min": 0, "max": 20, "step": 1},
    "Glucose": {"min": 0, "max": 300, "step": 1},
    "BloodPressure": {"min": 0, "max": 150, "step": 1},
    "SkinThickness": {"min": 0, "max": 100, "step": 1},
    "Insulin": {"min": 0, "max": 900, "step": 1},
    "BMI": {"min": 0.0, "max": 70.0, "step": 0.1},
    "Age": {"min": 1, "max": 120, "step": 1},
}


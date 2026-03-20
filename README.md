# Chronic Disease (Diabetes) Prediction – Streamlit ML App

A clean, modular final-year project that predicts **diabetes risk** using machine learning and a modern Streamlit UI.

## Features

- **Modular codebase**: separated UI, preprocessing, training, and prediction
- **3 ML models**: Logistic Regression, Random Forest, SVM (best selected by **accuracy**)
- **Preprocessing pipeline**: missing-value imputation + scaling
- **Probability output**: risk percentage + Low/Medium/High risk level
- **Health suggestions**: personalized, rule-based guidance
- **Visualizations**: Glucose vs Outcome, BMI distribution, permutation feature importance
- **PDF report download**: patient inputs + prediction + suggestions

## Project Structure

- `app/`: Streamlit UI (`ui.py`, `pages.py`)
- `model/`: ML logic (`train.py`, `predict.py`, `preprocessing.py`, `model.pkl`)
- `model/artifacts/`: training outputs (`metrics.json`, `feature_importance.json`, `training_summary.json`)
- `utils/`: validation, charts, PDF, suggestions
- `data/`: dataset (`data/dataset.csv`)
- `config.py`: shared constants/paths
- `main.py`: Streamlit entrypoint

## How to Run (Windows / PowerShell)

```powershell
cd "C:\Users\ELCOT\Final-Year-Project\chronic-disease"
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

## Retrain Models (Optional)

```powershell
python model/train.py --retrain
```

## Disclaimer

This project is for educational use and decision support only. It is **not** a substitute for professional medical advice.


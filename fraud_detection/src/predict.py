import joblib
import numpy as np
import pandas as pd
import os
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", 'models')
_model_cache = {}

def load_model(model_name: str):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at {path}\n"
            "Run training first in notebooks/02_training.ipynb"
        )
    if model_name in _model_cache:
        return _model_cache[model_name]
    model = joblib.load(path)
    _model_cache[model_name] = model
    return model

def get_risk_level(prob: float) -> str:
    if prob >= 0.8:
        return "HIGH"
    elif prob >= 0.5:
        return "MEDIUM"
    elif prob >= 0.3:
        return "LOW"
    else:
        return "VERY LOW"

def predict_single(transaction: dict, model_name: str,
                   threshold: float = 0.5) -> dict:
    model = load_model(model_name)
    X     = pd.DataFrame([transaction])
    prob  = float(model.predict_proba(X)[0][1])
    pred  = int(prob >= threshold)

    return {
        "prediction" : pred,
        "label"      : "FRAUD" if pred == 1 else "LEGIT",  # ← add this
        "probability": round(prob, 4),
        "risk_level" : get_risk_level(prob)
    }

def predict_batch(df, model_name, threshold=0.5):

    model = load_model(model_name)
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    result_df = df.copy()
    result_df["probability"] = probs
    result_df["prediction"] = preds
    result_df["risk_level"] = [get_risk_level(p) for p in probs]
    
    return result_df
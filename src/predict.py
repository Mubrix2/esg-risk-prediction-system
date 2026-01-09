# src/predict.py

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "esg_pipeline.pkl")

# Load once (fast)
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
FEATURES = bundle["features"]

def predict_esg(input_data: dict):
    df = pd.DataFrame([input_data])

    # Ensure all expected features exist
    missing = set(FEATURES) - set(df.columns)
    for col in missing:
        df[col] = 0.0  # safe default

    # Drop extra columns
    df = df[FEATURES]

    prediction = pipeline.predict(df)[0]

    return {
        "ESG_score_prediction": round(float(prediction), 2)
    }

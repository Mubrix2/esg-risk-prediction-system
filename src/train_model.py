# src/train_model.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "esg_processed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "esg_pipeline.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- CONFIG ----------------
TARGET = "ESG_score"

# Columns we NEVER want in training
DROP_COLS = [
    "Company Name",
    "Identifier (RIC)",
    "Industry",
    "Date"
]

# ---------------- TRAIN ----------------
def train():
    df = pd.read_csv(DATA_PATH)

    # Drop non-numeric / identifier columns
    df = df.drop(columns=DROP_COLS, errors="ignore")

    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found in dataset")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Ensure only numeric columns remain
    X = X.select_dtypes(include="number")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")

    # 🔥 SAVE PIPELINE + FEATURE SCHEMA
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": X_train.columns.tolist()
        },
        MODEL_PATH
    )

    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()

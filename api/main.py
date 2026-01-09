from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_esg

app = FastAPI(title="ESG Risk Prediction API")

# ---------------- INPUT SCHEMA ----------------
class ESGRequest(BaseModel):
    Env_score: float
    Social_score: float
    Gov_score: float
    PE_RATIO: float
    FNCL_LVRG: float
    RETURN_ON_ASSET: float
    ASSET_GROWTH: float
    QUICK_RATIO: float
    BVPS: float

# ---------------- ROUTES ----------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: ESGRequest):
    return predict_esg(payload.model_dump())

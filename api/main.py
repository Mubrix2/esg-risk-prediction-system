from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_esg

app = FastAPI(title="ESG Risk Prediction API")


class ESGInput(BaseModel):
    Env_score: float
    Social_score: float
    Gov_score: float
    PE_RATIO: float
    FNCL_LVRG: float
    RETURN_ON_ASSET: float
    ASSET_GROWTH: float
    QUICK_RATIO: float
    BVPS: float
    Net_income: float
    Shares: float
    Market_cap: float
    Total_assets: float

@app.get("/")
def home():
    return {"message": "ESG Risk Prediction API is running"}


@app.post("/predict")
def predict(input_data: ESGInput):
    prediction = predict_esg(input_data.dict())
    return {"esg_risk_score": prediction}

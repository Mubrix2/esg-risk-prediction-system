def predict_esg(input_data: dict):
    df = pd.DataFrame([input_data])
    # Optional: reorder columns to match training
    df = df[[
        "Env_score", "Social_score", "Gov_score", "PE_RATIO", "FNCL_LVRG",
        "RETURN_ON_ASSET", "ASSET_GROWTH", "QUICK_RATIO", "BVPS",
        "Net_income", "Shares", "Market_cap", "Total_assets"
    ]]
    prediction = model.predict(df)[0]
    return float(prediction)
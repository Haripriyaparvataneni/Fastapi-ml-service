"""FastAPI service exposing a product demand-forecasting model."""

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model import get_model, get_model_info
from app.schemas import ForecastRequest, ForecastResponse

app = FastAPI(
    title="Demand Forecasting API",
    description="Portfolio demo: serves a demand-forecasting model trained on synthetic data.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        return get_model_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    info = get_model_info()
    features = pd.DataFrame([{
        "price": request.price,
        "day_of_week": request.day_of_week,
        "month": request.month,
        "promotion": int(request.promotion),
        "stock_on_hand": request.stock_on_hand,
    }], columns=info["features"])
    predicted = float(model.predict(features)[0])
    return ForecastResponse(
        predicted_demand=round(predicted, 2),
        model_version=info["version"],
    )

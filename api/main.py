from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_model, predict_price

app = FastAPI(
    title="House Price Prediction API",
    description="Simple API for estimating a property price from its attributes.",
    version="1.0.0",
)
model = None


class HouseFeatures(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(ge=1, le=20)
    bathrooms: int = Field(ge=1, le=10)
    stories: int = Field(ge=1, le=20)
    parking: int = Field(ge=0, le=10)
    mainroad: Literal["yes", "no"]
    guestroom: Literal["yes", "no"]
    basement: Literal["yes", "no"]
    hotwaterheating: Literal["yes", "no"]
    airconditioning: Literal["yes", "no"]
    prefarea: Literal["yes", "no"]
    furnishingstatus: Literal["furnished", "semi-furnished", "unfurnished"]


@app.on_event("startup")
def load_artifact():
    global model
    try:
        model = load_model()
    except FileNotFoundError:
        model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained. Run python -m src.train first.")
    try:
        return predict_price(model, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "house_price_pipeline.joblib"
FEATURES = [
    "area", "bedrooms", "bathrooms", "stories", "parking",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]
CATEGORIES = {
    "mainroad": {"yes", "no"},
    "guestroom": {"yes", "no"},
    "basement": {"yes", "no"},
    "hotwaterheating": {"yes", "no"},
    "airconditioning": {"yes", "no"},
    "prefarea": {"yes", "no"},
    "furnishingstatus": {"furnished", "semi-furnished", "unfurnished"},
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model is not trained. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def predict_price(model, house: dict) -> dict:
    for feature in ["area", "bedrooms", "bathrooms", "stories", "parking"]:
        if float(house[feature]) < 0:
            raise ValueError(f"{feature} cannot be negative.")
    for feature, allowed in CATEGORIES.items():
        if house[feature] not in allowed:
            raise ValueError(f"Invalid value for {feature}: {house[feature]}")
    row = pd.DataFrame([{feature: house[feature] for feature in FEATURES}], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    return {"predicted_price": max(0.0, prediction), "currency": "dataset units"}

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Housing.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "house_price_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_FEATURES = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame = frame[FEATURES + [TARGET]].copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[TARGET].isna().any() or (frame[TARGET] <= 0).any():
        raise ValueError("Target price must contain positive numeric values.")
    for column in NUMERIC_FEATURES:
        if (frame[column].dropna() < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    return frame


def build_pipeline(estimator) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def train(use_mlflow: bool = True) -> dict:
    data = load_dataset()
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    candidates = {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=250, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        fold_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        scores[name] = {
            "mean_cv_rmse": float(-fold_scores.mean()),
            "std_cv_rmse": float(fold_scores.std()),
        }
    best_name = min(scores, key=lambda name: scores[name]["mean_cv_rmse"])
    model = build_pipeline(candidates[best_name])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "mean_actual_price": float(y_test.mean()),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "selected_model": best_name,
        "dataset_rows": int(len(data)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "cross_validation": scores,
        "holdout_metrics": metrics,
        "target_unit": "dataset price units",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if use_mlflow:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("house-price-prediction")
        with mlflow.start_run(run_name=best_name):
            mlflow.log_params({
                "selected_model": best_name,
                "dataset_rows": len(data),
                "test_size": 0.2,
                "random_state": 42,
            })
            mlflow.log_metrics({
                f"cv_{name}_mean_rmse": values["mean_cv_rmse"]
                for name, values in scores.items()
            })
            mlflow.log_metrics(metrics)
            mlflow.log_dict(metadata, "evaluation.json")
            mlflow.sklearn.log_model(model, artifact_path="model")
    print(json.dumps(metadata, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare house price regression models.")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    train(use_mlflow=not args.no_mlflow)

# House Price Prediction

An end-to-end regression application that extends the original house price project with reproducible model training, cross-validation, experiment tracking, an interactive dashboard, a lightweight prediction API, and containerized execution.

## What is included

- Data validation and feature selection.
- Scikit-learn preprocessing pipeline for numeric and categorical features.
- Comparison of Ridge Regression, Random Forest, and Histogram Gradient Boosting.
- Five-fold cross-validation using RMSE.
- Holdout evaluation using MAE, RMSE, and R².
- MLflow tracking for experiments, metrics, metadata, and model artifacts.
- Streamlit property price estimator and dataset insights.
- Simple FastAPI health and prediction endpoints.
- Docker and Docker Compose.
- GitHub Actions workflow for dependency installation, module compilation, and model-training validation.

## Repository structure

```text
.
├── api/
│   └── main.py
├── src/
│   ├── inference.py
│   └── train.py
├── .github/workflows/ci.yml
├── app.py
├── houseprice.py
├── housepriceprediction.ipynb
├── Housing.csv
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

The original `houseprice.py`, notebook, and dataset are retained. The new end-to-end application is available through `app.py`.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and train:

```bash
pip install -r requirements.txt
python -m src.train
```

Training validates the CSV, compares candidate models with five-fold cross-validation on the training split, selects the model with the lowest mean CV RMSE, evaluates it on a held-out test split, and writes the fitted pipeline and evaluation metadata to `artifacts/`.

## Run the dashboard

```bash
streamlit run app.py
```

The dashboard supports property price estimates, dataset summaries, and model evaluation.

## Run the API

```bash
uvicorn api.main:app --reload
```

- Health: `http://127.0.0.1:8000/health`
- Interactive documentation: `http://127.0.0.1:8000/docs`

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "area": 5000,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "parking": 1,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "yes",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "prefarea": "yes",
    "furnishingstatus": "semi-furnished"
  }'
```

The response includes the predicted price and identifies the units as dataset units. The dataset does not establish a reliable currency or current market location, so the application does not label the result as INR or another currency.

## MLflow

Training runs are saved in the local `mlruns/` directory. Start the UI with:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://127.0.0.1:5000` to inspect model runs, parameters, metrics, and artifacts.

## Docker

Train the model first, then run:

```bash
docker compose up --build
```

- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

## Modeling approach

The target is `price`. Numeric property attributes are median-imputed and scaled. Categorical attributes are mode-imputed and one-hot encoded. These transformations are part of the scikit-learn pipeline, so they are fitted on training data only.

The training split is used for model selection through shuffled five-fold cross-validation. A separate holdout split is used for final evaluation. RMSE penalizes larger errors more heavily, MAE provides an average absolute error, and R² summarizes explained variance on the holdout set.

## Limitations

- The dataset is a small, static housing dataset and may not represent current market prices.
- The source data does not provide a verified location, currency, or time period for generalizing predictions.
- Model estimates are not property valuations or financial advice.
- Real deployment would require recent local market data, external validation, monitoring, and review of location-specific factors.

## Original work

The original `houseprice.py`, `housepriceprediction.ipynb`, and `Housing.csv` remain in the repository. The new pipeline and application are added alongside the original project.

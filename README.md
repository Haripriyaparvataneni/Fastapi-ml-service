# Demand Forecasting API (FastAPI)

Training a model in a notebook is one thing — this is the "okay, now serve it over an API" part. A small FastAPI service that loads a trained demand forecasting model and answers prediction requests.

The model is a `GradientBoostingRegressor` from scikit-learn, trained on data I generated in `scripts/train_model.py` (5,000 rows: price, day of week, month, promotion, stock on hand → demand). It's made-up data standing in for what real product demand would look like — there's no company data or code in here.

## Try it

```bash
pip install -r requirements.txt

python scripts/train_model.py    # trains and saves model.joblib (test MAE: ~10.46 units)
uvicorn app.main:app --reload    # start the API
pytest tests/ -v                 # run the tests
```

Or with Docker:

```bash
docker build -t demand-api .
docker run -p 8000:8000 demand-api
```

## Endpoints

| Method | Path          | What it does                         |
|--------|---------------|--------------------------------------|
| GET    | `/health`     | Liveness check                       |
| GET    | `/model-info` | Model version, features, where it came from |
| POST   | `/predict`    | Demand forecast for one product row  |

## Examples

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl http://localhost:8000/model-info
# {"version":"1.0.0","artifact_path":"model.joblib",
#  "model_type":"GradientBoostingRegressor",
#  "features":["price","day_of_week","month","promotion","stock_on_hand"],
#  "trained_on":"synthetic product-demand data (see scripts/train_model.py)"}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"price": 25.0, "day_of_week": 5, "month": 12, "promotion": true, "stock_on_hand": 200}'
# {"predicted_demand":173.82,"model_version":"1.0.0"}
```

Send something invalid, like a `day_of_week` of 9, and you get a 422 back with a Pydantic validation error instead of a nonsense prediction. That validation is half the reason for using FastAPI + Pydantic here.

## Project layout

```
fastapi-ml-service/
├── app/
│   ├── main.py        # FastAPI app: /health, /model-info, /predict
│   ├── model.py       # loads the model artifact + metadata
│   └── schemas.py     # ForecastRequest / ForecastResponse (Pydantic)
├── scripts/
│   └── train_model.py # generates the data and trains the model
├── tests/
│   └── test_api.py    # pytest suite using TestClient
├── Dockerfile
├── requirements.txt
└── .gitignore
```

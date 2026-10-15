"""API tests using FastAPI's TestClient (no live server needed)."""

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent

# Train the model artifact once before the app module imports it.
subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "train_model.py")],
    check=True,
    capture_output=True,
)

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_happy_path():
    payload = {
        "price": 25.0,
        "day_of_week": 5,  # Saturday
        "month": 12,
        "promotion": True,
        "stock_on_hand": 200,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_demand"] >= 0
    assert body["model_version"] == "1.0.0"


def test_predict_validation_error():
    # day_of_week out of range (0-6) must return 422
    payload = {
        "price": 25.0,
        "day_of_week": 9,
        "month": 12,
        "promotion": False,
        "stock_on_hand": 200,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_field():
    # missing required field `price` must return 422
    payload = {
        "day_of_week": 3,
        "month": 6,
        "promotion": False,
        "stock_on_hand": 100,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_model_info():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "1.0.0"
    assert "features" in body

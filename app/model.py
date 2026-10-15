"""Load (and cache) the trained demand-forecasting model artifact."""

from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
MODEL_VERSION = "1.0.0"

_model = None


def get_model():
    """Return the trained model, loading it from disk on first use."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_PATH}. "
                "Run `python scripts/train_model.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_model_info() -> dict:
    """Metadata about the served model artifact."""
    return {
        "version": MODEL_VERSION,
        "artifact_path": str(MODEL_PATH.name),
        "model_type": type(get_model()).__name__,
        "features": ["price", "day_of_week", "month", "promotion", "stock_on_hand"],
        "trained_on": "synthetic product-demand data (see scripts/train_model.py)",
    }

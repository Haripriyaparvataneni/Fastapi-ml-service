"""Train a demand-forecasting regressor on synthetic data and save it.

The dataset is fully synthetic — generated here with numpy — and only
stands in for what real product-demand data would look like.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
RANDOM_STATE = 42


def generate_data(n_samples: int = 5_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create synthetic product-demand rows."""
    rng = np.random.default_rng(seed)

    price = rng.uniform(5, 100, n_samples)
    day_of_week = rng.integers(0, 7, n_samples)
    month = rng.integers(1, 13, n_samples)
    promotion = rng.integers(0, 2, n_samples)
    stock_on_hand = rng.integers(0, 500, n_samples)

    # Plausible demand curve: promotions and weekends lift demand,
    # higher prices lower it, plus seasonality and noise.
    weekend = (day_of_week >= 5).astype(float)
    seasonal = np.sin(2 * np.pi * month / 12)
    demand = (
        120
        - 0.9 * price
        + 40 * promotion
        + 25 * weekend
        + 15 * seasonal
        + 0.05 * stock_on_hand
        + rng.normal(0, 12, n_samples)
    )
    demand = np.clip(demand, 0, None)

    return pd.DataFrame({
        "price": price,
        "day_of_week": day_of_week,
        "month": month,
        "promotion": promotion,
        "stock_on_hand": stock_on_hand,
        "demand": demand,
    })


def main() -> None:
    df = generate_data()
    X = df.drop(columns=["demand"])
    y = df["demand"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Test MAE: {mae:.2f} units")

    joblib.dump(model, ARTIFACT_PATH)
    print(f"Saved model to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()

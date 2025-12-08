from __future__ import annotations

import json

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config, data_prep, features


def train_frequency_model() -> None:
    df = data_prep.load_clean()
    X, y = features.split_features_target_frequency(df)

    # Drop rows where mapping failed
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    preprocessor = features.get_preprocessor()

    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", reg),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(math.sqrt(mse))


    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "frequency_model.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "mae": mae,
        "rmse": rmse,
    }

    config.REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.REPORTS_METRICS_DIR / "frequency_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved frequency model to {model_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    train_frequency_model()

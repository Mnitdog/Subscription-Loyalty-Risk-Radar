from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)
import math

from sklearn.model_selection import train_test_split

from . import config, data_prep, features


def evaluate_subscription() -> dict:
    df = data_prep.load_clean()
    X, y = features.split_features_target_subscription(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    model_path = config.MODELS_DIR / "subscription_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Subscription model not found at {model_path}")
    model = joblib.load(model_path)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }

    return metrics


def evaluate_frequency() -> dict:
    df = data_prep.load_clean()
    X, y = features.split_features_target_frequency(df)
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    model_path = config.MODELS_DIR / "frequency_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Frequency model not found at {model_path}")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(math.sqrt(mse))

    metrics = {
        "mae": mae,
        "rmse": rmse,
    }


    return metrics


def main() -> None:
    sub_metrics = evaluate_subscription()
    freq_metrics = evaluate_frequency()

    config.REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = config.REPORTS_METRICS_DIR / "subscription_metrics_eval.json"
    freq_path = config.REPORTS_METRICS_DIR / "frequency_metrics_eval.json"

    with sub_path.open("w", encoding="utf-8") as f:
        json.dump(sub_metrics, f, indent=2)
    with freq_path.open("w", encoding="utf-8") as f:
        json.dump(freq_metrics, f, indent=2)

    print("Subscription metrics:", sub_metrics)
    print("Frequency metrics:", freq_metrics)


if __name__ == "__main__":
    main()

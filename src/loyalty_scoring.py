from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from . import config, data_prep, features


@dataclass
class LoyaltyModels:
    subscription_model: object
    frequency_model: object


def load_models() -> LoyaltyModels:
    sub_path = config.MODELS_DIR / "subscription_model.joblib"
    freq_path = config.MODELS_DIR / "frequency_model.joblib"

    if not sub_path.exists() or not freq_path.exists():
        raise FileNotFoundError(
            "Models not found. Train them first with `python -m src.cli train-all`."
        )

    subscription_model = joblib.load(sub_path)
    frequency_model = joblib.load(freq_path)
    return LoyaltyModels(subscription_model=subscription_model, frequency_model=frequency_model)


def compute_loyalty_for_row(row: pd.Series, models: LoyaltyModels) -> Dict[str, float]:
    """Compute subscription probability, frequency score and loyalty risk for a single row."""
    row_df = row.to_frame().T
    X = row_df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]

    p_subscribe = float(models.subscription_model.predict_proba(X)[:, 1][0])
    freq_score_pred = float(models.frequency_model.predict(X)[0])

    # Clamp frequency score to [1, 7]
    freq_score_clamped = float(min(max(freq_score_pred, 1.0), 7.0))

    freq_component = freq_score_clamped / 7.0
    loyalty_index = 0.6 * p_subscribe + 0.4 * freq_component
    loyalty_index = float(max(min(loyalty_index, 1.0), 0.0))

    loyalty_risk = float((1.0 - loyalty_index) * 100.0)

    return {
        "p_subscribe": p_subscribe,
        "predicted_frequency_score": freq_score_clamped,
        "loyalty_index": loyalty_index,
        "loyalty_risk": loyalty_risk,
    }


def score_all_customers(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None:
        df = data_prep.load_clean()

    models = load_models()

    scores = []
    for _, row in df.iterrows():
        metrics = compute_loyalty_for_row(row, models)
        scores.append(metrics)

    scores_df = pd.DataFrame(scores)
    result = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    return result

from __future__ import annotations

from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from . import config, data_prep, features


def _get_feature_names(model) -> List[str]:
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    return list(feature_names)


def global_feature_importance_subscription(top_n: int = 20) -> pd.DataFrame:
    df = data_prep.load_clean()
    X, y = features.split_features_target_subscription(df)

    model_path = config.MODELS_DIR / "subscription_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Subscription model not found. Train it first.")

    model = joblib.load(model_path)
    feature_names = _get_feature_names(model)
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.fit_transform(X)  # only for shape; not used directly

    clf = model.named_steps["model"]
    importances = clf.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return importance_df.head(top_n)


def global_feature_importance_frequency(top_n: int = 20) -> pd.DataFrame:
    df = data_prep.load_clean()
    X, y = features.split_features_target_frequency(df)
    mask = y.notna()
    X = X[mask]

    model_path = config.MODELS_DIR / "frequency_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Frequency model not found. Train it first.")

    model = joblib.load(model_path)
    feature_names = _get_feature_names(model)
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.fit_transform(X)

    reg = model.named_steps["model"]
    importances = reg.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return importance_df.head(top_n)

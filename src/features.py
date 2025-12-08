from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def get_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that preprocesses numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_FEATURES),
            ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def split_features_target_subscription(df: pd.DataFrame):
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = (df[config.TARGET_SUBSCRIPTION].astype(str).str.lower() == "yes").astype(int)
    return X, y


def split_features_target_frequency(df: pd.DataFrame):
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET_FREQUENCY].map(config.FREQUENCY_MAPPING)
    return X, y

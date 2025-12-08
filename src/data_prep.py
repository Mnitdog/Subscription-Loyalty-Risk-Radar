from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Load the raw Kaggle CSV file."""
    path = config.DATA_RAW
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at: {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and type normalization.

    This function keeps the dataset very close to the original,
    mainly ensuring consistent dtypes and stripping whitespace.
    """
    df = df.copy()

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Ensure numeric dtypes where appropriate
    numeric_cols = [
        "Age",
        "Purchase Amount (USD)",
        "Previous Purchases",
        "Review Rating",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # <-- FIXED HERE

    # Drop rows with critical missing values
    df = df.dropna(
        subset=[
            config.TARGET_SUBSCRIPTION,
            config.TARGET_FREQUENCY,
            "Age",
            "Purchase Amount (USD)",
        ]
    )

    return df



def load_clean() -> pd.DataFrame:
    """Load a cleaned version of the dataset, creating it if needed."""
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = config.DATA_PROCESSED_DIR / "shopping_behavior_clean.parquet"

    if clean_path.exists():
        return pd.read_parquet(clean_path)

    df_raw = load_raw()
    df_clean = clean_data(df_raw)
    df_clean.to_parquet(clean_path, index=False)
    return df_clean

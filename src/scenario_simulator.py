from __future__ import annotations

from typing import Dict

import pandas as pd

from . import config, data_prep
from .loyalty_scoring import LoyaltyModels, compute_loyalty_for_row, load_models


def simulate_scenario(
    base_row: pd.Series, changes: Dict[str, object], models: LoyaltyModels | None = None
) -> Dict[str, Dict[str, float]]:
    """Simulate a what-if scenario by applying `changes` to a row.

    Returns a dict with `before` and `after` loyalty metrics.
    """
    if models is None:
        models = load_models()

    before = compute_loyalty_for_row(base_row, models)

    modified_row = base_row.copy()
    for key, value in changes.items():
        if key in modified_row.index:
            modified_row[key] = value

    after = compute_loyalty_for_row(modified_row, models)

    deltas = {k: after[k] - before[k] for k in before.keys()}
    return {"before": before, "after": after, "delta": deltas}

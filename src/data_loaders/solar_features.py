from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


SOLAR_ACCUMULATED_DEFAULT: List[str] = ["VAR169", "VAR175", "VAR178", "VAR228"]


def decumulate_series_with_resets(s: pd.Series) -> pd.Series:
    """
    Convert an accumulated field into increments with reset handling.

    Rule:
      increment[t] = s[t] - s[t-1]
      if increment[t] < 0 (reset), set increment[t] = s[t]
      clip negatives to 0 to avoid small numerical artifacts
    """
    diff = s.diff()
    diff = diff.where(diff >= 0, s)
    return diff.clip(lower=0)


def decumulate_solar_accumulators(
    df: pd.DataFrame,
    cols: Iterable[str] = SOLAR_ACCUMULATED_DEFAULT,
    group_level: int = 0,
) -> pd.DataFrame:
    """
    Apply decumulation to the specified columns of a MultiIndex dataframe.
    Expected index levels: [ZONEID, timestamp].
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        # Use transform() to preserve the original MultiIndex shape exactly.
        out[col] = out.groupby(level=group_level, sort=False)[col].transform(decumulate_series_with_resets)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional helper: add cyclical time features from the timestamp index.
    Useful for transformers (known-future covariates).
    """
    if not isinstance(df.index, pd.MultiIndex) or "timestamp" not in df.index.names:
        raise ValueError("Expected MultiIndex with level named 'timestamp'")

    ts = df.index.get_level_values("timestamp")
    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month

    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return out

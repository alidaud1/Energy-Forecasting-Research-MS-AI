"""Make the package importable when pytest is run from anywhere in the repo."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS  # noqa: E402

LEVELS = np.arange(1, 100) / 100.0


@pytest.fixture
def normal_bundle():
    """Build a bundle whose marginals are exactly Normal(mu, sigma).

    Fusion is then checkable against closed-form results instead of against
    whatever the code happens to produce.
    """
    def build(source, mu, sigma, *, n=24, start="2010-01-01", unit="MW",
              physical_min=None, physical_max=None, actual=None, metadata=None):
        frame = pd.DataFrame({"TIMESTAMP": pd.date_range(start, periods=n, freq="h")})
        grid = norm.ppf(LEVELS, loc=mu, scale=sigma)
        frame = pd.concat([frame, pd.DataFrame(
            np.tile(grid, (n, 1)), columns=list(QUANTILE_COLUMNS))], axis=1)
        actual_column = None
        if actual is not None:
            actual_column = f"actual_{source.upper()}"
            frame[actual_column] = np.asarray(actual, dtype=float)
        return ForecastBundle(source, frame, unit, source, "24h", actual_column,
                              metadata or {}, physical_min, physical_max)
    return build


@pytest.fixture
def solar_shaped():
    """A bundle with a realistic diurnal shape: zero at night, peak at noon."""
    def build(peak_hour=12, n=720, start="2014-06-01 01:00", capacity=60.0, unit="MW"):
        stamps = pd.date_range(start, periods=n, freq="h")
        shape = np.clip(np.cos((stamps.hour.to_numpy() - peak_hour) / 24 * 2 * np.pi), 0, None) ** 2
        median = shape * 0.75 * capacity
        spread = np.where(shape > 0, 0.10 * capacity, 0.0)
        grid = median[:, None] + spread[:, None] * norm.ppf(LEVELS)[None, :]
        grid = np.clip(grid, 0.0, capacity)
        frame = pd.DataFrame({"TIMESTAMP": stamps})
        frame = pd.concat([frame, pd.DataFrame(grid, columns=list(QUANTILE_COLUMNS))], axis=1)
        return ForecastBundle("solar", frame, unit, "zone-1", "24h", None,
                              {"capacity_mw": capacity}, 0.0, capacity)
    return build

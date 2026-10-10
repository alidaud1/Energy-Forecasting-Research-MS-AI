"""Fast tests for the measured-data forecasting baseline."""

import numpy as np
import pandas as pd

from elsevier_paper.experiments.run_opsd_baseline import (
    _features, _quantiles_by_hour, _score,
)


def test_forecast_features_use_completed_intervals_at_least_25_hours_old():
    n = 200
    data = pd.DataFrame({
        "timestamp_utc": pd.date_range("2017-01-01", periods=n, freq="h", tz="UTC")
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "load_mw": np.arange(n, dtype=float) + 100,
        "solar_mw": np.arange(n, dtype=float) % 10,
        "solar_capacity_mw": np.full(n, 20.0),
    })
    features, _, _ = _features(data)
    assert features.loc[100, "load_lag25"] == data.loc[75, "load_mw"]
    assert pd.isna(features.loc[100, "load_lag169"])  # unavailable at issue time
    assert features.loc[180, "solar_ratio_lag169"] == data.loc[11, "solar_mw"] / 20
    assert features.loc[100, "capacity_lag25"] == data.loc[75, "solar_capacity_mw"]


def test_calibration_quantiles_are_monotone_and_bounded():
    cal_hour = np.tile(np.arange(24), 60)
    cal_point = np.full(len(cal_hour), 5.0)
    cal_actual = cal_point + np.tile(np.linspace(-3, 3, 60), 24)
    result = _quantiles_by_hour(np.array([0.0, 5.0]), cal_point, cal_actual,
                                np.array([0, 1]), cal_hour, low=0.0, high=10.0)
    assert result.shape == (2, 99)
    assert np.all(result >= 0)
    assert np.all(result <= 10)
    assert np.all(np.diff(result, axis=1) >= 0)


def test_score_reports_perfect_median_mae():
    actual = np.array([10.0, 20.0])
    quantiles = np.repeat(actual[:, None], 99, axis=1)
    result = _score(actual, quantiles)
    assert result["n_hours"] == 2
    assert result["median_mae_mw"] == 0.0
    assert result["mean_pinball_loss_mw"] == 0.0
    assert result["coverage_q05_q95"] == 1.0

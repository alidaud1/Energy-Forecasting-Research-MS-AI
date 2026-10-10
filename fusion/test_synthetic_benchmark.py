"""The joint benchmark must calibrate dependence before its evaluation hours."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion.synthetic_benchmark import create_benchmark
from fusion.joint_pipeline import run_joint_pipeline


def test_synthetic_benchmark_uses_temporal_holdout(tmp_path):
    result = create_benchmark(tmp_path, days=4, calibration_days=2,
                              seed=123, n_scenarios=100)
    evaluation = result["evaluation"]
    assert evaluation["n_timestamps"] == 48
    assert evaluation["dependence_uses_evaluation_actuals"] is False
    assert pd.Timestamp(evaluation["calibration_end"]) < pd.Timestamp(
        evaluation["evaluation_start"])

    report = json.loads((tmp_path / "joint_run/joint_manifest.json").read_text())
    diagnostics = report["fusion_diagnostics"]
    assert diagnostics["dependence_source"] == "stated by caller"
    assert diagnostics["rank_correlation"] == pytest.approx(
        evaluation["calibration_rank_correlation"])

    forecast = pd.read_csv(tmp_path / "joint_run/net_load_forecast.csv")
    actuals = pd.read_csv(tmp_path / "synthetic_actuals.csv")
    assert len(forecast) == 48
    assert forecast["TIMESTAMP"].tolist() == actuals["TIMESTAMP"].iloc[48:].tolist()
    assert np.isfinite(forecast.filter(regex=r"^q\d\d$").to_numpy()).all()


def test_holdout_metrics_are_reproducible(tmp_path):
    first = create_benchmark(tmp_path / "one", days=4, calibration_days=2,
                             seed=99, n_scenarios=100)
    second = create_benchmark(tmp_path / "two", days=4, calibration_days=2,
                              seed=99, n_scenarios=100)
    for key in ("mean_pinball_loss_mw", "coverage_q05_q95",
                "mean_interval_width_q05_q95_mw", "calibration_rank_correlation"):
        assert first["evaluation"][key] == second["evaluation"][key]


def test_held_out_actuals_cannot_change_fused_forecast(tmp_path):
    result = create_benchmark(tmp_path, days=4, calibration_days=2,
                              seed=77, n_scenarios=100)
    original = pd.read_csv(tmp_path / "joint_run/net_load_forecast.csv")
    load_path = tmp_path / "load_predictions_test.csv"
    solar_path = tmp_path / "solar_index_test.csv"
    load = pd.read_csv(load_path)
    solar = pd.read_csv(solar_path)
    load["actual_LOAD"] += 1000.0
    solar["POWER"] = 0.0
    load.to_csv(load_path, index=False)
    solar.to_csv(solar_path, index=False)
    rerun = run_joint_pipeline(
        load_path, tmp_path / "solar_predictions_test.csv", solar_path,
        tmp_path / "actuals_changed", solar_capacity_mw=60.0,
        n_scenarios=100, seed=77,
        dependence=result["evaluation"]["calibration_rank_correlation"],
        solar_capacity_basis="ASSUMED: installed capacity of simulated grid",
    )
    changed = pd.read_csv(rerun["forecast_path"])
    pd.testing.assert_frame_equal(original, changed)


@pytest.mark.parametrize("kwargs", [
    {"days": 1},
    {"days": 4, "calibration_days": 0},
    {"days": 4, "calibration_days": 4},
    {"days": 4, "n_scenarios": 1},
])
def test_invalid_split_or_scenario_count_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        create_benchmark(Path(tmp_path), **kwargs)

import json
from pathlib import Path

import numpy as np

from elsevier_paper.experiments.replay_solar_checkpoints import json_safe, metrics


def test_parameter_export_is_strict_json_and_preserves_missing_sentinel():
    value = json_safe({"missing": np.nan, "quantile_alpha": np.float32(.5)})
    assert json.loads(json.dumps(value, allow_nan=False))["missing"] == "nan"
    assert value["quantile_alpha"] == .5


def test_capacity_fraction_metrics_are_explicit_percentages():
    q = np.tile(np.arange(1, 100)/100, (2, 1))
    measured = metrics(np.array([.5, .5]), q)
    assert measured["mae_percent_capacity"] == 0
    assert measured["coverage_q10_q90_percent"] == 100
    np.testing.assert_allclose(measured["width_percent_capacity"], 80)


def test_completed_keyed_replays_and_recorded_ensemble_are_verified():
    from elsevier_paper.experiments.summarize_checkpoint_replay import run
    report = run()
    assert len(report["models"]) == 9
    for name, row in report["models"].items():
        assert row["splits"]["test"]["metrics"]["pinball_percent_capacity"] > 0
        if name in ("XGBoost", "LightGBM", "CatBoost"):
            assert row["splits"]["test"]["archive_comparison"]["max_absolute_difference"] == 0


def test_tcn_cpu_setting_is_documented_with_measured_arithmetic_control():
    path = Path(__file__).resolve().parent / "solar_checkpoint_replay"
    control = json.loads((path / "tcn_cpu_numerics.json").read_text())
    config = json.loads((path / "TCN/configuration.json").read_text())
    assert config["cpu_flush_float32_subnormals"] is True
    assert control["subnormal_weight_count"] > 0
    assert control["max_absolute_output_difference"] <= 1e-7

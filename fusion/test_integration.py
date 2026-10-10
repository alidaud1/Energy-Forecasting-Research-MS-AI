"""End-to-end checks against the real load and solar artifacts."""

import json
from pathlib import Path

import numpy as np
import pytest

from fusion.adapters import load_bundle, solar_bundle
from fusion.alignment import align_cyclically, describe_overlap
from fusion.fusion_agent import fuse_forecasts
from fusion.joint_pipeline import run_joint_pipeline

ROOT = Path(__file__).resolve().parents[1]
LOAD = ROOT / "agentic/outputs/test_predictions_monotone/GBR_predictions_monotone.csv"
SOLAR = ROOT / "agentic_solar/solar-forecasting-GEFCom2014/pipeline/outputs/predictions/Ensemble_predictions.csv"
INDEX = ROOT / "agentic_solar/solar-forecasting-GEFCom2014/pipeline/outputs/splits/CatBoost/test.csv"
CAPACITY = 60.0

pytestmark = pytest.mark.skipif(
    not (LOAD.is_file() and SOLAR.is_file() and INDEX.is_file()),
    reason="real forecast artifacts are not present")


@pytest.fixture(scope="module")
def bundles():
    load = load_bundle(LOAD)
    solar = solar_bundle(SOLAR, timestamps_path=INDEX, capacity_mw=CAPACITY)
    return load, align_cyclically(load, solar)


def test_real_tracks_do_not_overlap():
    load = load_bundle(LOAD)
    solar = solar_bundle(SOLAR, timestamps_path=INDEX, capacity_mw=CAPACITY)
    assert describe_overlap(load, solar)["shared_steps"] == 0


def test_alignment_places_solar_in_daylight(bundles):
    _, solar = bundles
    by_hour = solar.frame.groupby(solar.timestamps().hour)["q50"].mean()
    assert by_hour.idxmax() in (11, 12, 13)
    assert by_hour[[0, 1, 2, 3, 22, 23]].max() < 0.01 * CAPACITY


def test_solar_is_a_material_share_of_net_load(bundles):
    """The unit bug made solar 0.06% of load; it must now actually bite."""
    load, solar = bundles
    result = fuse_forecasts(load, solar, n_scenarios=400, seed=42)
    share = result.solar_scenarios.mean() / result.load_scenarios.mean()
    assert 0.02 < share < 0.5
    midday = result.forecast["TIMESTAMP"].dt.hour == 12
    assert result.forecast.loc[midday, "q50"].mean() < load.frame.loc[midday.to_numpy(), "q50"].mean()


def test_real_scenarios_respect_bounds(bundles):
    load, solar = bundles
    result = fuse_forecasts(load, solar, n_scenarios=400, seed=42)
    assert result.solar_scenarios.min() >= 0.0
    assert result.solar_scenarios.max() <= CAPACITY
    assert result.load_scenarios.min() >= 0.0


def test_pipeline_writes_a_manifest_that_states_its_assumptions(tmp_path):
    report = run_joint_pipeline(LOAD, SOLAR, INDEX, tmp_path,
                                solar_capacity_mw=CAPACITY, n_scenarios=200, seed=42)
    manifest = json.loads(Path(report["manifest_path"]).read_text())
    assumptions = manifest["assumptions"]
    assert assumptions["solar_capacity_mw"] == CAPACITY
    assert "ASSUMED" in assumptions["solar_capacity_basis"]
    assert "mod 720" in assumptions["alignment_rule"]
    assert assumptions["solar_local_offset_hours"] == 10
    assert "cyclically" in assumptions["caveat"]
    assert manifest["calendar_overlap"]["shared_steps"] == 0
    assert Path(report["forecast_path"]).is_file()
    assert json.loads(Path(report["decision_report_path"]).read_text())["summary"]


def test_pipeline_is_reproducible(tmp_path):
    kw = dict(solar_capacity_mw=CAPACITY, n_scenarios=200, seed=7)
    a = run_joint_pipeline(LOAD, SOLAR, INDEX, tmp_path / "a", **kw)["fusion"]
    b = run_joint_pipeline(LOAD, SOLAR, INDEX, tmp_path / "b", **kw)["fusion"]
    np.testing.assert_array_equal(a.forecast.to_numpy(), b.forecast.to_numpy())


def test_cli_runs_end_to_end(tmp_path):
    from fusion.run_existing import main
    assert main(["--output", str(tmp_path), "--scenarios", "100"]) == 0
    assert (tmp_path / "net_load_forecast.csv").is_file()
    manifest = json.loads((tmp_path / "joint_manifest.json").read_text())
    assert manifest["fusion_diagnostics"]["solar_capacity_mw"] == 60.0

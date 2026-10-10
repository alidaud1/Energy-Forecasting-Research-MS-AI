import numpy as np
import pandas as pd
import pytest

from fusion.adapters import load_bundle, solar_bundle
from fusion.forecast_contract import ForecastContractError, QUANTILE_COLUMNS


@pytest.fixture
def artifacts(tmp_path):
    """A miniature two-zone solar artifact plus its index file."""
    n_hours, zones = 12, (1, 2)
    stamps = pd.date_range("2014-06-01 01:00", periods=n_hours, freq="h")
    rows, index_rows = [], []
    for zone in zones:
        for i, stamp in enumerate(stamps):
            fraction = (i + 1) / 100.0
            rows.append({q: (fraction + j * 1e-5) * zone for j, q in enumerate(QUANTILE_COLUMNS)})
            index_rows.append({"ZONEID": zone, "TIMESTAMP": stamp, "POWER": fraction * zone})
    predictions = tmp_path / "Model_predictions.csv"
    index = tmp_path / "test.csv"
    pd.DataFrame(rows).to_csv(predictions, index=False)
    pd.DataFrame(index_rows).to_csv(index, index=False)
    return predictions, index


def test_capacity_is_mandatory_and_applied(artifacts):
    predictions, index = artifacts
    bundle = solar_bundle(predictions, timestamps_path=index, capacity_mw=100.0)
    assert bundle.unit == "MW"
    assert bundle.frame["q01"].iloc[0] == pytest.approx(1.0)      # 0.01 fraction * 100 MW
    assert bundle.physical_max == 100.0
    assert bundle.metadata["capacity_mw"] == 100.0
    assert "ASSUMED" in bundle.metadata["capacity_basis"]


def test_capacity_scales_output_linearly(artifacts):
    predictions, index = artifacts
    small = solar_bundle(predictions, timestamps_path=index, capacity_mw=50.0)
    large = solar_bundle(predictions, timestamps_path=index, capacity_mw=100.0)
    np.testing.assert_allclose(large.quantiles(), small.quantiles() * 2)


@pytest.mark.parametrize("capacity", [0, -5, float("nan"), float("inf")])
def test_invalid_capacity_is_rejected(artifacts, capacity):
    predictions, index = artifacts
    with pytest.raises(ForecastContractError, match="capacity_mw"):
        solar_bundle(predictions, timestamps_path=index, capacity_mw=capacity)


def test_double_conversion_is_refused(artifacts, tmp_path):
    """Feeding already-converted MW back in would multiply by capacity twice."""
    predictions, index = artifacts
    frame = pd.read_csv(predictions) * 100.0
    already_mw = tmp_path / "mw_predictions.csv"
    frame.to_csv(already_mw, index=False)
    with pytest.raises(ForecastContractError, match="capacity fractions"):
        solar_bundle(already_mw, timestamps_path=index, capacity_mw=100.0)


def test_zone_selection(artifacts):
    predictions, index = artifacts
    one = solar_bundle(predictions, timestamps_path=index, capacity_mw=100.0, location_value=1)
    two = solar_bundle(predictions, timestamps_path=index, capacity_mw=100.0, location_value=2)
    assert len(one.frame) == len(two.frame) == 12
    np.testing.assert_allclose(two.quantiles(), one.quantiles() * 2)
    with pytest.raises(ForecastContractError, match="no rows"):
        solar_bundle(predictions, timestamps_path=index, capacity_mw=100.0, location_value=99)


def test_row_count_mismatch_is_refused(artifacts, tmp_path):
    """The row-zip is only valid if both artifacts have the same length."""
    predictions, index = artifacts
    short = tmp_path / "short.csv"
    pd.read_csv(index).iloc[:5].to_csv(short, index=False)
    with pytest.raises(ForecastContractError, match="not row-aligned"):
        solar_bundle(predictions, timestamps_path=short, capacity_mw=100.0)


def test_actuals_are_attached_when_present(artifacts, tmp_path):
    predictions, index = artifacts
    bundle = solar_bundle(predictions, timestamps_path=index, capacity_mw=100.0)
    assert bundle.actual_column == "actual_SOLAR"
    assert bundle.residuals() is not None
    without = tmp_path / "no_power.csv"
    pd.read_csv(index).drop(columns="POWER").to_csv(without, index=False)
    assert solar_bundle(predictions, timestamps_path=without, capacity_mw=100.0).actual_column is None


def test_missing_files_and_columns(artifacts, tmp_path):
    predictions, index = artifacts
    with pytest.raises(ForecastContractError, match="does not exist"):
        solar_bundle(tmp_path / "nope.csv", timestamps_path=index, capacity_mw=100.0)
    bad = tmp_path / "bad_index.csv"
    pd.read_csv(index).rename(columns={"TIMESTAMP": "T"}).to_csv(bad, index=False)
    with pytest.raises(ForecastContractError, match="lacks"):
        solar_bundle(predictions, timestamps_path=bad, capacity_mw=100.0)


def test_load_bundle_has_actuals_and_a_floor(tmp_path):
    columns = {"TIMESTAMP": pd.date_range("2010-01-01", periods=6, freq="h")}
    columns.update({q: [100.0 + i] * 6 for i, q in enumerate(QUANTILE_COLUMNS)})
    columns["actual_LOAD"] = [140.0] * 6
    frame = pd.DataFrame(columns)
    path = tmp_path / "load.csv"
    frame.to_csv(path, index=False)
    bundle = load_bundle(path)
    assert bundle.unit == "MW" and bundle.physical_min == 0.0
    assert bundle.actual_column == "actual_LOAD"
    assert bundle.residuals() is not None

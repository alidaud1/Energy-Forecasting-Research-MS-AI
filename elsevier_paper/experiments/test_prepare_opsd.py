"""Offline integrity tests for the measured OPSD preparation boundary."""

import pandas as pd
import pytest

from elsevier_paper.experiments.prepare_opsd import CAPACITY, LOAD, SOLAR, prepare


def _source(path):
    frame = pd.DataFrame({
        "utc_timestamp": ["2017-01-01T00:00:00Z", "2017-01-01T01:00:00Z",
                          "2017-01-01T02:00:00Z"],
        "cet_cest_timestamp": ["2017-01-01T01:00:00+0100",
                               "2017-01-01T02:00:00+0100",
                               "2017-01-01T03:00:00+0100"],
        LOAD: [100.0, 110.0, 120.0],
        SOLAR: [0.0, 5.0, 8.0],
        CAPACITY: [10.0, 10.0, None],
    })
    frame.to_csv(path, index=False)
    return frame


def test_prepares_measured_series_without_imputing_actuals(tmp_path):
    source, output = tmp_path / "source.csv", tmp_path / "clean.csv"
    _source(source)
    report = prepare(source, output, expected_rows=3)
    cleaned = pd.read_csv(output)
    assert report["rows"] == 3
    assert report["capacity_forward_filled_hours"] == 1
    assert cleaned["solar_capacity_mw"].tolist() == [10.0, 10.0, 10.0]
    assert cleaned["load_mw"].tolist() == [100.0, 110.0, 120.0]
    assert cleaned["solar_mw"].tolist() == [0.0, 5.0, 8.0]
    assert (tmp_path / "clean_manifest.json").is_file()


@pytest.mark.parametrize("damage,match", [
    ("gap", "consecutive"),
    ("duplicate", "duplicated"),
    ("missing_solar", "missing observations"),
    ("over_capacity", "exceeds"),
])
def test_rejects_invalid_measured_data(tmp_path, damage, match):
    source, output = tmp_path / "source.csv", tmp_path / "clean.csv"
    frame = _source(source)
    if damage == "gap":
        frame.loc[2, "utc_timestamp"] = "2017-01-01T03:00:00Z"
    elif damage == "duplicate":
        frame.loc[2, "utc_timestamp"] = frame.loc[1, "utc_timestamp"]
    elif damage == "missing_solar":
        frame.loc[1, SOLAR] = None
    else:
        frame.loc[1, SOLAR] = 11.0
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match=match):
        prepare(source, output, expected_rows=3)
    assert not output.exists()


def test_rejects_partial_download(tmp_path):
    source = tmp_path / "source.csv.partial"
    _source(source)
    with pytest.raises(ValueError, match="partial"):
        prepare(source, tmp_path / "clean.csv", expected_rows=3)

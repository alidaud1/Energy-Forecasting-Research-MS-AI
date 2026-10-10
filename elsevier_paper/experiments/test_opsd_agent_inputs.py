"""Integration checks for the isolated measured-agent input hand-off."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "opsd_agent_port"
MEASURED = ROOT / "opsd_source" / "de_2017_2019_clean.csv"


def test_split_contract_and_timestamp_alignment():
    manifest = json.loads((INPUTS / "input_manifest.json").read_text())
    assert manifest["split_rows"] == {"train": 12934, "calibration": 4417, "test": 8760}
    assert manifest["model_training_completed"] is False
    assert manifest["weather_forecasts_available"] is False
    for split, expected in manifest["split_rows"].items():
        load = pd.read_csv(INPUTS / f"load_{split}.csv")
        solar = pd.read_csv(INPUTS / f"solar_{split}.csv")
        assert len(load) == len(solar) == expected
        assert load.TIMESTAMP.equals(solar.TIMESTAMP)
        assert load.AREA.eq("DE").all() and solar.AREA.eq("DE").all()
        assert "LOAD" not in solar.columns and "POWER" not in load.columns
        assert not any(c.startswith("VAR") or c.startswith("w")
                       for c in load.columns if c not in {"TIMESTAMP", "AREA"})
        assert solar.POWER.between(0, 1).all()


def test_first_test_forecast_uses_only_completed_older_intervals():
    raw = pd.read_csv(MEASURED)
    load = pd.read_csv(INPUTS / "load_test.csv", nrows=1).iloc[0]
    solar = pd.read_csv(INPUTS / "solar_test.csv", nrows=1).iloc[0]
    target_index = raw.index[raw.timestamp_utc == load.TIMESTAMP][0]
    assert solar.TIMESTAMP == load.TIMESTAMP
    assert np.isclose(load.load_lag25, raw.loc[target_index - 25, "load_mw"])
    assert np.isclose(solar.capacity_lag25,
                      raw.loc[target_index - 25, "solar_capacity_mw"])
    assert np.isclose(solar.solar_ratio_lag25,
                      raw.loc[target_index - 25, "solar_mw"] /
                      raw.loc[target_index - 25, "solar_capacity_mw"])
    assert np.isclose(solar.POWER,
                      raw.loc[target_index, "solar_mw"] /
                      raw.loc[target_index - 25, "solar_capacity_mw"])

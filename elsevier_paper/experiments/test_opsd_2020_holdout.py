"""Guard the new holdout and post-hoc calibration contracts."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from elsevier_paper.experiments.calibrate_opsd_native_subset import quantile_offsets
from elsevier_paper.experiments.prepare_opsd_2020_holdout import prepare
from elsevier_paper.experiments.verify_opsd_native_2020 import verify


def test_offsets_match_empirical_quantiles():
    actual = np.arange(150, dtype=float)
    forecast = np.tile(np.arange(99, dtype=float), (150, 1))
    offsets = quantile_offsets(actual, forecast)
    for i, q in enumerate(np.arange(1, 100) / 100):
        assert offsets[i] == pytest.approx(np.quantile(actual - forecast[:, i], q))


def test_offsets_reject_bad_inputs():
    with pytest.raises(ValueError, match="invalid marginal"):
        quantile_offsets(np.ones(10), np.ones((10, 99)))
    with pytest.raises(ValueError, match="invalid marginal"):
        quantile_offsets(np.ones(150), np.ones((150, 98)))


def test_holdout_has_no_imputed_targets(tmp_path):
    root = tmp_path
    # The real source contract deliberately requires the complete interval.
    history = pd.DataFrame({"timestamp_utc": ["2019-12-31T22:00:00Z"],
                            "solar_capacity_mw": [50_508.0]})
    raw = pd.DataFrame({"utc_timestamp": ["2019-12-31T23:00:00Z"],
                        "cet_cest_timestamp": ["2020-01-01T00:00:00+0100"],
                        "DE_load_actual_entsoe_transparency": [np.nan],
                        "DE_solar_generation_actual": [0],
                        "DE_solar_capacity": [np.nan]})
    history.to_csv(root / "history.csv", index=False)
    raw.to_csv(root / "raw.csv", index=False)
    with pytest.raises(ValueError, match="complete Jan–Jun"):
        prepare(root / "history.csv", root / "raw.csv", root / "out")


def test_saved_2020_scores_recompute():
    result_dir = Path(__file__).resolve().parent / "opsd_native_2020"
    result = verify(result_dir)
    assert result["verified_hours"] == 4367
    assert result["decision_reports"] == 12

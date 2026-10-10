import numpy as np
import pandas as pd
import pytest

from elsevier_paper.experiments.audit_solar_statistics import hac_mean, holm_adjust, hourly_zone_mean, shared_day_ci


def test_hac_matches_reference_formula():
    from scipy.stats import norm
    d = np.random.default_rng(1).normal(.02, .1, 720)
    centered = d - d.mean()
    lrv = centered @ centered / len(d)
    for lag in range(1, 25):
        lrv += 2 * (1 - lag / 25) * (centered[lag:] @ centered[:-lag]) / len(d)
    se = np.sqrt(lrv / len(d))
    measured = hac_mean(d, 24)
    assert measured["standard_error"] == pytest.approx(se)
    assert measured["p_value"] == pytest.approx(2 * norm.sf(abs(d.mean() / se)))


def test_common_hour_aggregation_is_order_invariant():
    index = pd.DataFrame({"ZONEID": np.repeat([1, 2, 3], 720),
                          "TIMESTAMP": np.tile(pd.date_range("2014-06-01 01:00", periods=720, freq="h"), 3)})
    losses = np.tile(np.arange(720), 3) + np.repeat([0, 1, 2], 720)
    np.testing.assert_allclose(hourly_zone_mean(index, losses), np.arange(720) + 1)
    shuffled = np.random.default_rng(1).permutation(2160)
    np.testing.assert_allclose(hourly_zone_mean(index.iloc[shuffled], losses[shuffled]), np.arange(720) + 1)


def test_shared_day_ci_is_reproducible():
    values = np.repeat(np.arange(30), 24).astype(float)
    assert shared_day_ci(values) == shared_day_ci(values)
    lo, hi = shared_day_ci(values)
    assert lo < values.mean() < hi


def test_holm_corrects_family_and_preserves_order():
    np.testing.assert_allclose(holm_adjust([.04, .001, .02]), [.04, .003, .04])


@pytest.mark.parametrize("lag", [-1, 720])
def test_invalid_hac_lag_is_rejected(lag):
    with pytest.raises(ValueError):
        hac_mean(np.ones(720), lag)

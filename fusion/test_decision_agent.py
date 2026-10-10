import numpy as np
import pytest

from fusion.decision_agent import make_decisions
from fusion.forecast_contract import ForecastContractError
from fusion.fusion_agent import fuse_forecasts

CAPACITY = 60.0


def fused(load_mu=150, load_sd=10, solar_mu=30, solar_sd=5, **kw):
    from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS
    import pandas as pd
    from scipy.stats import norm
    levels = np.arange(1, 100) / 100.0

    def bundle(source, mu, sd, meta=None, pmin=None, pmax=None):
        n = 24
        frame = pd.DataFrame({"TIMESTAMP": pd.date_range("2010-01-01", periods=n, freq="h")})
        grid = np.clip(norm.ppf(levels, mu, sd), 0, None)
        frame = pd.concat([frame, pd.DataFrame(np.tile(grid, (n, 1)),
                                               columns=list(QUANTILE_COLUMNS))], axis=1)
        return ForecastBundle(source, frame, "MW", source, "24h", None, meta or {}, pmin, pmax)

    return fuse_forecasts(bundle("load", load_mu, load_sd, pmin=0.0),
                          bundle("solar", solar_mu, solar_sd,
                                 {"capacity_mw": CAPACITY}, 0.0, CAPACITY),
                          n_scenarios=kw.pop("n_scenarios", 4000), seed=kw.pop("seed", 1))


def test_summary_is_auditable():
    decision = make_decisions(fused())
    assert decision.summary["n_timestamps"] == 24
    assert decision.summary["unit"] == "MW"
    assert decision.summary["solar_capacity_mw"] == CAPACITY
    assert decision.recommendations


def test_reserve_is_the_gap_to_the_chosen_quantile():
    result = fused()
    decision = make_decisions(result, reserve_quantile="q95")
    expected = (result.forecast["q95"] - result.forecast["q50"]).max()
    assert decision.summary["reserve_requirement_mw"] == pytest.approx(expected, rel=1e-9)
    assert decision.summary["reserve_quantile"] == "q95"


def test_reserve_margin_is_added():
    base = make_decisions(fused()).summary["reserve_requirement_mw"]
    padded = make_decisions(fused(), reserve_margin_mw=7.5).summary["reserve_requirement_mw"]
    assert padded == pytest.approx(base + 7.5, rel=1e-9)


def test_low_solar_tracks_generation_not_dispersion():
    """The original bug: the flag keyed off solar's own median, so a wide but
    sunny forecast tripped it while a tight, genuinely dark one did not."""
    confident_and_sunny = make_decisions(fused(solar_mu=45, solar_sd=1)).summary
    genuinely_dark = make_decisions(fused(solar_mu=3, solar_sd=1)).summary
    assert confident_and_sunny["low_solar_indicator"] is False
    assert genuinely_dark["low_solar_indicator"] is True


def test_low_solar_is_measured_against_installed_capacity():
    result = fused(solar_mu=6, solar_sd=1)
    assert make_decisions(result, low_solar_threshold=0.20).summary["low_solar_indicator"] is True
    assert make_decisions(result, low_solar_threshold=0.05).summary["low_solar_indicator"] is False


def test_night_hours_do_not_drag_the_capacity_factor(solar_shaped, normal_bundle):
    """Overnight zeros are excluded, so the factor describes daylight output."""
    result = fuse_forecasts(normal_bundle("load", 150, 10, n=720, start="2014-06-01 01:00"),
                            solar_shaped(peak_hour=12), n_scenarios=1500, seed=2)
    summary = make_decisions(result).summary
    assert 0 < summary["daylight_hours_in_window"] < 720
    assert summary["solar_daylight_capacity_factor"] > 0.2


def test_ramp_and_uncertainty_metrics():
    summary = make_decisions(fused()).summary
    assert summary["max_upward_ramp_mw"] >= 0 >= summary["max_downward_ramp_mw"]
    assert summary["peak_net_load_mw"] >= summary["min_net_load_mw"]
    assert 0 <= summary["load_uncertainty_ratio"] < 1


def test_high_load_uncertainty_flag_responds_to_spread():
    assert "high_load_uncertainty" not in make_decisions(fused(load_sd=1)).risk_flags
    assert "high_load_uncertainty" in make_decisions(fused(load_sd=60)).risk_flags


def test_recommendations_never_empty():
    decision = make_decisions(fused(load_sd=1, solar_mu=45, solar_sd=1))
    assert decision.recommendations


@pytest.mark.parametrize("kwargs", [
    dict(reserve_quantile="q50"), dict(reserve_quantile="q10"), dict(reserve_quantile="nope"),
    dict(reserve_margin_mw=-1), dict(low_solar_threshold=0), dict(low_solar_threshold=1),
    dict(load_uncertainty_threshold=1.5), dict(daylight_fraction=0),
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ForecastContractError):
        make_decisions(fused(), **kwargs)

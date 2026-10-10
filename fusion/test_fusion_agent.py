import numpy as np
import pytest

from fusion.forecast_contract import ForecastContractError
from fusion.fusion_agent import _gaussian_copula_parameter, fuse_forecasts


def test_net_load_is_load_minus_solar(normal_bundle):
    result = fuse_forecasts(normal_bundle("load", 150, 10),
                            normal_bundle("solar", 30, 5), n_scenarios=20000, seed=1)
    assert result.net_load_scenarios.mean() == pytest.approx(120, abs=0.5)
    assert result.forecast["q50"].mean() == pytest.approx(120, abs=0.5)


def test_monte_carlo_not_same_rank_subtraction(normal_bundle):
    """Independent margins must ADD variance, not cancel it.

    Subtracting same-rank quantiles would give sd = 10 - 5 = 5; correct
    independent sampling gives sqrt(10^2 + 5^2) = 11.18.
    """
    result = fuse_forecasts(normal_bundle("load", 150, 10),
                            normal_bundle("solar", 30, 5), n_scenarios=40000, seed=2)
    assert result.net_load_scenarios.std() == pytest.approx(np.hypot(10, 5), rel=0.03)
    assert result.net_load_scenarios.std() > 10.0        # never below either margin


def test_dependence_narrows_the_net_spread(normal_bundle):
    kwargs = dict(n_scenarios=40000, seed=3)
    load, solar = normal_bundle("load", 150, 10), normal_bundle("solar", 30, 5)
    independent = fuse_forecasts(load, solar, **kwargs)
    correlated = fuse_forecasts(load, solar, dependence=0.9, **kwargs)
    assert correlated.net_load_scenarios.std() < independent.net_load_scenarios.std()
    assert correlated.diagnostics["dependence_source"] == "stated by caller"


def test_rank_correlation_is_converted_for_the_copula():
    assert _gaussian_copula_parameter(0.0) == 0.0
    assert _gaussian_copula_parameter(1.0) == 0.999      # clipped off the singularity
    assert _gaussian_copula_parameter(-1.0) == -0.999
    assert _gaussian_copula_parameter(0.5) == pytest.approx(2 * np.sin(np.pi / 12))
    assert _gaussian_copula_parameter(0.5) != 0.5            # the original bug
    assert _gaussian_copula_parameter(-0.5) == pytest.approx(-2 * np.sin(np.pi / 12))


def test_dependence_estimated_from_actuals(normal_bundle):
    rng = np.random.default_rng(0)
    load_actual = 150 + rng.normal(size=24) * 10
    result = fuse_forecasts(
        normal_bundle("load", 150, 10, actual=load_actual),
        normal_bundle("solar", 30, 5, actual=30 + (load_actual - 150) * 0.5),
        n_scenarios=5000, seed=4)
    assert result.diagnostics["dependence_source"] == "estimated from bundle actuals"
    assert result.diagnostics["rank_correlation"] > 0.5


def test_independence_is_reported_when_no_actuals(normal_bundle):
    result = fuse_forecasts(normal_bundle("load", 150, 10),
                            normal_bundle("solar", 30, 5), n_scenarios=100)
    assert result.diagnostics["rank_correlation"] == 0.0
    assert "assumed independent" in result.diagnostics["dependence_source"]


def test_scenarios_respect_physical_bounds(normal_bundle):
    """Tail extrapolation must not invent negative or over-capacity generation."""
    result = fuse_forecasts(
        normal_bundle("load", 150, 10, physical_min=0.0),
        normal_bundle("solar", 5, 8, physical_min=0.0, physical_max=20.0),
        n_scenarios=40000, seed=5)
    assert result.solar_scenarios.min() >= 0.0
    assert result.solar_scenarios.max() <= 20.0
    assert result.load_scenarios.min() >= 0.0


def test_unbounded_solar_would_go_negative(normal_bundle):
    """Guards the guard: without bounds the extrapolation really does misbehave."""
    result = fuse_forecasts(normal_bundle("load", 150, 10),
                            normal_bundle("solar", 5, 8), n_scenarios=40000, seed=5)
    assert result.solar_scenarios.min() < 0.0


def test_net_quantiles_never_cross(normal_bundle):
    result = fuse_forecasts(normal_bundle("load", 150, 10),
                            normal_bundle("solar", 30, 5), n_scenarios=5000, seed=6)
    values = result.forecast.drop(columns="TIMESTAMP").to_numpy()
    assert (np.diff(values, axis=1) >= -1e-9).all()


def test_no_overlap_points_at_the_alignment_helper(normal_bundle):
    load = normal_bundle("load", 150, 10, start="2010-01-01")
    solar = normal_bundle("solar", 30, 5, start="2014-06-01")
    with pytest.raises(ForecastContractError, match="align_cyclically"):
        fuse_forecasts(load, solar)


def test_unit_conversion(normal_bundle):
    """A kW solar bundle must be scaled, not silently treated as MW."""
    mw = fuse_forecasts(normal_bundle("load", 150, 10),
                        normal_bundle("solar", 30, 5), n_scenarios=8000, seed=7)
    kw = fuse_forecasts(normal_bundle("load", 150, 10),
                        normal_bundle("solar", 30000, 5000, unit="kW"),
                        n_scenarios=8000, seed=7)
    assert kw.net_load_scenarios.mean() == pytest.approx(mw.net_load_scenarios.mean(), rel=1e-6)


@pytest.mark.parametrize("kwargs, match", [
    (dict(n_scenarios=1), "at least 2"),
    (dict(dependence=1.5), "rank correlation"),
    (dict(dependence=float("nan")), "rank correlation"),
])
def test_fusion_validates_inputs(normal_bundle, kwargs, match):
    with pytest.raises(ForecastContractError, match=match):
        fuse_forecasts(normal_bundle("load", 150, 10), normal_bundle("solar", 30, 5), **kwargs)


def test_horizon_mismatch_is_rejected(normal_bundle):
    solar = normal_bundle("solar", 30, 5)
    other = solar.__class__(solar.source, solar.frame, solar.unit, solar.location, "48h")
    with pytest.raises(ForecastContractError, match="horizons"):
        fuse_forecasts(normal_bundle("load", 150, 10), other)


def test_results_are_reproducible(normal_bundle):
    args = (normal_bundle("load", 150, 10), normal_bundle("solar", 30, 5))
    a = fuse_forecasts(*args, n_scenarios=2000, seed=11)
    b = fuse_forecasts(*args, n_scenarios=2000, seed=11)
    np.testing.assert_array_equal(a.net_load_scenarios, b.net_load_scenarios)

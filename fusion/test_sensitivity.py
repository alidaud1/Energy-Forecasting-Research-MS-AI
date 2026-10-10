"""Checks on the shortcut-versus-sampling comparison quoted in the thesis.

Section 4.7.1 rests on two claims about arithmetic rather than on the particular
data: that subtracting matching quantiles produces the load interval less the
solar interval, and that sampling instead combines the two spreads.  Both are
checkable in closed form against Normal marginals, which is what these do.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion.fusion_agent import fuse_forecasts
from fusion.sensitivity import HIGH, LOW, _mean_width, same_rank_width


def test_same_rank_width_is_the_difference_of_the_marginal_widths(normal_bundle):
    """The shortcut subtracts spreads: 10-sigma load less 5-sigma solar."""
    load = normal_bundle("load", 150, 10)
    solar = normal_bundle("solar", 30, 5)

    load_width = _mean_width(load.frame)
    solar_width = _mean_width(solar.frame)

    assert same_rank_width(load, solar) == pytest.approx(load_width - solar_width)


def test_sampling_widens_where_subtraction_narrows(normal_bundle):
    """Independent sampling combines the two spreads in quadrature instead.

    For Normal marginals the sampled 90 per cent width is 2 * 1.6449 * sqrt(10^2
    + 5^2), against the subtracted width of 2 * 1.6449 * (10 - 5).  The point of
    the test is the direction: introducing solar uncertainty must not make the
    reported net-load interval narrower than the load interval alone.
    """
    load = normal_bundle("load", 150, 10)
    solar = normal_bundle("solar", 30, 5)

    fused = fuse_forecasts(load, solar, n_scenarios=40000, seed=7, dependence=0.0)
    sampled = _mean_width(fused.forecast)
    subtracted = same_rank_width(load, solar)

    span = 2 * 1.6448536269514722  # q95 - q05 of a standard Normal
    assert sampled == pytest.approx(span * np.hypot(10, 5), rel=0.02)
    assert subtracted == pytest.approx(span * (10 - 5), rel=1e-6)
    assert sampled > _mean_width(load.frame) > subtracted


def test_shortcut_approaches_the_perfect_dependence_limit(normal_bundle):
    """Sampling under near-perfect rank dependence converges on the shortcut.

    This is the claim that makes same-rank subtraction a dependence assumption
    rather than a neutral simplification, so it is asserted rather than argued.
    """
    load = normal_bundle("load", 150, 10)
    solar = normal_bundle("solar", 30, 5)
    subtracted = same_rank_width(load, solar)

    widths = [_mean_width(fuse_forecasts(load, solar, n_scenarios=20000, seed=3,
                                         dependence=rho).forecast)
              for rho in (0.0, 0.5, 0.99)]

    assert widths[0] > widths[1] > widths[2]
    assert widths[2] == pytest.approx(subtracted, rel=0.05)


def test_band_is_the_one_the_reserve_discussion_uses():
    """The reported band must stay q05-q95; Section 5.3 quotes it as such."""
    assert (LOW, HIGH) == ("q05", "q95")

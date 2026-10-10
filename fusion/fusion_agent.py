"""Deterministic joint load/solar scenario fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import spearmanr

from .forecast_contract import ForecastBundle, ForecastContractError, QUANTILE_COLUMNS

_LEVELS = np.arange(1, 100, dtype=float) / 100.0
_UNIT_SCALE = {"W": 1.0, "kW": 1_000.0, "MW": 1_000_000.0, "GW": 1_000_000_000.0}


@dataclass(frozen=True)
class FusionResult:
    """Output of fusion, including scenarios for downstream decisions."""

    forecast: pd.DataFrame
    load_scenarios: np.ndarray
    solar_scenarios: np.ndarray
    net_load_scenarios: np.ndarray
    correlation: float
    diagnostics: dict


def _to_mw(values: np.ndarray, unit: str) -> np.ndarray:
    if unit not in _UNIT_SCALE:
        raise ForecastContractError(f"unsupported energy unit: {unit}")
    return values * (_UNIT_SCALE[unit] / _UNIT_SCALE["MW"])


def _rank_dependence(load_residuals, solar_residuals) -> float:
    """Spearman rank correlation of the two residual series.

    Rank correlation, not Pearson: the copula step below is driven by ranks, and
    a Pearson coefficient fed into it would misstate the dependence whenever
    either margin is skewed -- which solar always is, being a bounded series with
    a large mass at zero overnight.
    """
    left, right = np.asarray(load_residuals, float), np.asarray(solar_residuals, float)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right) or len(left) < 3:
        raise ForecastContractError("residual arrays must be equal 1-D arrays with at least 3 values")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ForecastContractError("residual arrays must be finite")
    rho = float(spearmanr(left, right).statistic)
    return 0.0 if not np.isfinite(rho) else float(np.clip(rho, -0.99, 0.99))


def _gaussian_copula_parameter(rank_correlation: float) -> float:
    """Convert a Spearman rank correlation to a Gaussian copula correlation.

    For the Gaussian copula, rho_spearman = (6/pi) * arcsin(rho/2), so the
    inverse is rho = 2 * sin(pi * rho_spearman / 6).  Using the rank correlation
    directly as the copula parameter -- as the first revision did -- overstates
    dependence by up to about 4.5%.
    """
    return float(np.clip(2.0 * np.sin(np.pi * rank_correlation / 6.0), -0.999, 0.999))


def _clamp(values: np.ndarray, bundle: ForecastBundle) -> np.ndarray:
    """Hold sampled scenarios inside the domain's physical limits.

    Sampling below q01 or above q99 extrapolates off the end of the quantile
    grid, which for solar produced about 1% negative generation.  Clamping is
    applied to the sampled scenarios rather than to the quantile grid so the
    tails keep their spread instead of being truncated to the q01/q99 envelope.
    """
    low = -np.inf if bundle.physical_min is None else _to_mw(np.asarray(bundle.physical_min, float), bundle.unit)
    high = np.inf if bundle.physical_max is None else _to_mw(np.asarray(bundle.physical_max, float), bundle.unit)
    return np.clip(values, low, high)


def fuse_forecasts(load: ForecastBundle, solar: ForecastBundle, *,
                   n_scenarios: int = 1000, seed: int = 42,
                   load_residuals=None, solar_residuals=None,
                   dependence: float | None = None,
                   output_location: str = "grid") -> FusionResult:
    """Align two bundles and produce correlated net-load scenarios.

    Marginal distributions are reconstructed by interpolating each forecast's
    Q01--Q99 values, then sampled by Monte Carlo -- never by subtracting
    same-rank quantiles, which would assume perfect dependence.  Gaussian-copula
    uniforms impose the estimated residual dependence, sampled scenarios are
    clamped to each domain's physical limits, and the net-load quantiles are read
    back off the scenario ensemble (``np.quantile`` returns them in order, so
    they cannot cross).

    Dependence is taken from ``dependence`` when given (a stated Spearman
    correlation), otherwise estimated from the residual arrays, otherwise from
    the bundles' own actuals when both carry them.  Independence is used only as
    a last resort and is reported as such in ``diagnostics``.
    """
    if n_scenarios < 2:
        raise ForecastContractError("n_scenarios must be at least 2")
    if load.horizon != solar.horizon:
        raise ForecastContractError("load and solar horizons must match")

    if dependence is not None:
        if not np.isfinite(dependence) or not -1.0 < float(dependence) < 1.0:
            raise ForecastContractError("dependence must be a rank correlation in (-1, 1)")
        rank_corr, source = float(dependence), "stated by caller"
    elif load_residuals is not None and solar_residuals is not None:
        rank_corr, source = _rank_dependence(load_residuals, solar_residuals), "estimated from supplied residuals"
    elif load.actual_column is not None and solar.actual_column is not None \
            and len(load.frame) == len(solar.frame):
        rank_corr = _rank_dependence(load.residuals(), solar.residuals())
        source = "estimated from bundle actuals"
        if solar.metadata.get("alignment") == "cyclical":
            source += " (solar cyclically aligned; dependence is not physical)"
    else:
        rank_corr, source = 0.0, "assumed independent (no actuals available)"
    corr = _gaussian_copula_parameter(rank_corr)

    left = load.frame[["TIMESTAMP", *QUANTILE_COLUMNS]].copy()
    right = solar.frame[["TIMESTAMP", *QUANTILE_COLUMNS]].copy()
    merged = left.merge(right, on="TIMESTAMP", how="inner", suffixes=("_load", "_solar"))
    if merged.empty:
        raise ForecastContractError(
            "load and solar forecasts have no common timestamps "
            f"(load {load.timestamps().min()}..{load.timestamps().max()}, "
            f"solar {solar.timestamps().min()}..{solar.timestamps().max()}). "
            "The GEFCom load and solar tracks do not overlap; apply "
            "fusion.alignment.align_cyclically before fusing.")
    load_q = _to_mw(merged[[f"{q}_load" for q in QUANTILE_COLUMNS]].to_numpy(float), load.unit)
    solar_q = _to_mw(merged[[f"{q}_solar" for q in QUANTILE_COLUMNS]].to_numpy(float), solar.unit)

    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal((2, n_scenarios, len(merged)))
    z2 = corr * z1 + np.sqrt(1.0 - corr * corr) * z2
    load_u, solar_u = ndtr(z1), ndtr(z2)

    # np.interp operates row-wise via a small vectorized search/interpolation.
    def sample(marginals, uniforms):
        idx = np.clip(np.searchsorted(_LEVELS, uniforms, side="left"), 1, 98)
        lo, hi = _LEVELS[idx - 1], _LEVELS[idx]
        a = np.take_along_axis(marginals.T, idx - 1, axis=0)
        b = np.take_along_axis(marginals.T, idx, axis=0)
        return a + (b - a) * (uniforms - lo) / (hi - lo)

    load_s = _clamp(sample(load_q, load_u), load)
    solar_s = _clamp(sample(solar_q, solar_u), solar)
    net_s = load_s - solar_s
    quantiles = np.quantile(net_s, _LEVELS, axis=0).T
    out = pd.DataFrame({"TIMESTAMP": merged["TIMESTAMP"].to_numpy()})
    for i, q in enumerate(QUANTILE_COLUMNS):
        out[q] = quantiles[:, i]
    return FusionResult(out, load_s, solar_s, net_s, corr, {
        "n_scenarios": n_scenarios, "n_timestamps": len(out),
        "rank_correlation": rank_corr,
        "copula_correlation": corr,
        "dependence_source": source,
        "unit": "MW", "location": output_location, "seed": seed,
        "solar_capacity_mw": solar.metadata.get("capacity_mw"),
        "solar_clamped_to": [solar.physical_min, solar.physical_max],
        "alignment": solar.metadata.get("alignment", "calendar overlap"),
        "alignment_rule": solar.metadata.get("alignment_rule"),
    })

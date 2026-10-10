"""Deterministic operational decision support for fused net-load forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forecast_contract import ForecastContractError, QUANTILE_COLUMNS
from .fusion_agent import FusionResult


@dataclass(frozen=True)
class DecisionResult:
    """Auditable summary and recommendations derived from fused scenarios."""

    summary: dict
    recommendations: list[str]
    risk_flags: list[str]


def _daylight_mask(solar_median: np.ndarray, capacity_mw: float | None,
                   daylight_fraction: float) -> np.ndarray:
    """Hours in which the array is generating at all.

    Overnight hours are legitimately zero and must be excluded before judging
    whether solar output is poor; averaging them in guarantees a low number
    every day and tells an operator nothing.
    """
    reference = capacity_mw if capacity_mw else float(np.max(solar_median))
    if reference <= 0:
        return np.zeros_like(solar_median, dtype=bool)
    return solar_median > daylight_fraction * reference


def make_decisions(result: FusionResult, *, reserve_quantile: str = "q95",
                   reserve_margin_mw: float = 0.0,
                   load_uncertainty_threshold: float = 0.15,
                   low_solar_threshold: float = 0.20,
                   daylight_fraction: float = 0.01) -> DecisionResult:
    """Calculate transparent reserve and dispatch guidance from scenarios.

    ``low_solar_threshold`` is read against the array's installed capacity, not
    against solar's own median.  The earlier revision compared solar's 10th
    percentile to a fraction of its own median, which is a statement about
    dispersion rather than generation: it fired on a wide-but-sunny forecast and
    stayed silent on a tight, genuinely dark one.
    """
    if reserve_quantile not in QUANTILE_COLUMNS or int(reserve_quantile[1:]) <= 50:
        raise ForecastContractError("reserve_quantile must be q51--q99")
    if reserve_margin_mw < 0 or not 0 < load_uncertainty_threshold < 1 \
            or not 0 < low_solar_threshold < 1 or not 0 < daylight_fraction < 1:
        raise ForecastContractError("decision thresholds are outside valid ranges")

    q = result.forecast
    median = q["q50"].to_numpy(float)
    upper = q[reserve_quantile].to_numpy(float)
    reserve = np.maximum(0.0, upper - median + reserve_margin_mw)
    ramps = np.diff(median, prepend=median[0])
    upper_ramps = np.diff(upper, prepend=upper[0])

    solar_median = np.median(result.solar_scenarios, axis=0)
    load_median = np.median(result.load_scenarios, axis=0)
    capacity = result.diagnostics.get("solar_capacity_mw")
    daylight = _daylight_mask(solar_median, capacity, daylight_fraction)
    reference = float(capacity) if capacity else float(np.max(solar_median) or 1.0)

    # Judge generation only where the sun is up, and against installed capacity.
    if daylight.any():
        solar_p10_daylight = float(np.quantile(result.solar_scenarios[:, daylight], 0.10, axis=0).mean())
        capacity_factor = float(solar_median[daylight].mean() / reference)
        low_solar = solar_p10_daylight < low_solar_threshold * reference
    else:
        solar_p10_daylight, capacity_factor, low_solar = 0.0, 0.0, False

    load_p90 = np.quantile(result.load_scenarios, 0.90, axis=0)
    load_uncertainty = float(np.max(load_p90 - load_median) / max(np.mean(load_median), 1e-9))

    # Net-load ramping is the operational reason this fusion exists: solar
    # suppresses midday demand, so the steepest ramp is what must be covered.
    hours = pd.DatetimeIndex(q["TIMESTAMP"]).hour.to_numpy() if "TIMESTAMP" in q else np.zeros(len(q), int)
    window = min(3, max(1, len(median) - 1))
    rolling = median[window:] - median[:-window] if len(median) > window else np.zeros(1)

    flags, recommendations = [], []
    if low_solar:
        flags.append("low_solar_generation")
        recommendations.append("Schedule additional conventional generation or storage for low solar scenarios.")
    if load_uncertainty > load_uncertainty_threshold:
        flags.append("high_load_uncertainty")
        recommendations.append("Maintain extra upward reserve for high-demand scenarios.")
    if float(np.max(rolling)) > 0.10 * max(float(np.max(median)), 1e-9):
        flags.append("steep_net_load_ramp")
        recommendations.append(
            f"Provision flexible capacity for a {float(np.max(rolling)):.1f} MW "
            f"net-load ramp over {window} hour(s).")
    if float(np.max(reserve)) > 0:
        recommendations.append("Allocate upward reserve at least equal to the forecast uncertainty margin.")
    if not recommendations:
        recommendations.append("Operate the planned dispatch and monitor the fused forecast.")

    return DecisionResult({
        "location": result.diagnostics.get("location", "grid"),
        "unit": result.diagnostics.get("unit", "MW"),
        "n_timestamps": len(q),
        "peak_net_load_mw": float(np.max(median)),
        "peak_net_load_hour": int(hours[int(np.argmax(median))]),
        "min_net_load_mw": float(np.min(median)),
        "min_net_load_hour": int(hours[int(np.argmin(median))]),
        "peak_net_load_q95_mw": float(np.max(q[reserve_quantile])),
        "max_upward_ramp_mw": float(np.max(ramps)),
        "max_downward_ramp_mw": float(np.min(ramps)),
        f"max_{window}h_net_load_ramp_mw": float(np.max(rolling)),
        "max_q95_upward_ramp_mw": float(np.max(upper_ramps)),
        "reserve_requirement_mw": float(np.max(reserve)),
        "mean_reserve_requirement_mw": float(np.mean(reserve)),
        "solar_capacity_mw": capacity,
        "solar_daylight_capacity_factor": capacity_factor,
        "solar_p10_daylight_mw": solar_p10_daylight,
        "daylight_hours_in_window": int(daylight.sum()),
        "low_solar_indicator": bool(low_solar),
        "load_uncertainty_ratio": load_uncertainty,
        "reserve_quantile": reserve_quantile,
    }, recommendations, flags)

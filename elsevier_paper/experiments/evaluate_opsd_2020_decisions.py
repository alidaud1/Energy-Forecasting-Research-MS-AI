"""Check measurable 2020 peak/ramp/tail proxies from saved fusion forecasts.

This is not reserve-action or dispatch validation: the public OPSD measurements
contain no operator reserve requirements, activation or cost labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _variant(actual: pd.DataFrame, forecast: pd.DataFrame,
             decisions: list[dict]) -> dict:
    stamps = pd.to_datetime(actual.TIMESTAMP, utc=True)
    if not stamps.equals(pd.to_datetime(forecast.TIMESTAMP, utc=True)):
        raise ValueError("forecast and actual timestamps differ")
    y = actual.net_load_actual_mw.to_numpy(float)
    median = forecast.q50.to_numpy(float)
    upper = forecast.q95.to_numpy(float)
    lower = forecast.q05.to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(median).all() or \
            not np.isfinite(upper).all() or not np.isfinite(lower).all() or \
            (upper < median).any() or (lower > median).any():
        raise ValueError("invalid decision proxy data")
    months = stamps.dt.tz_convert("Europe/Berlin").dt.month.to_numpy()
    if sorted(np.unique(months).tolist()) != list(range(1, 7)) or \
            [d["month"] for d in decisions] != list(range(1, 7)):
        raise ValueError("incomplete six-month decision reports")
    peak_errors, max_ramp_errors = [], []
    hourly_ramp_errors, three_hour_ramp_errors = [], []
    monthly = []
    for m in range(1, 7):
        take = months == m
        observed, predicted, q95, q05 = y[take], median[take], upper[take], lower[take]
        if len(observed) < 24:
            raise ValueError("month has fewer than 24 observations")
        reported = decisions[m - 1]["summary"]
        pred_peak = float(np.max(predicted))
        pred_max_ramp = float(np.max(predicted[3:] - predicted[:-3]))
        pred_reserve = float(np.max(q95 - predicted))
        if not np.isclose(reported["peak_net_load_mw"], pred_peak, atol=1e-9) or \
                not np.isclose(reported["max_3h_net_load_ramp_mw"], pred_max_ramp,
                               atol=1e-9) or \
                not np.isclose(reported["reserve_requirement_mw"], pred_reserve,
                               atol=1e-9):
            raise ValueError(f"month {m}: saved decision summary is not derived from forecasts")
        actual_peak = float(np.max(observed))
        actual_max_ramp = float(np.max(observed[3:] - observed[:-3]))
        peak_errors.append(abs(pred_peak - actual_peak))
        max_ramp_errors.append(abs(pred_max_ramp - actual_max_ramp))
        hourly_ramp_errors.extend(np.abs(np.diff(predicted) - np.diff(observed)))
        three_hour_ramp_errors.extend(
            np.abs((predicted[3:] - predicted[:-3]) -
                   (observed[3:] - observed[:-3])))
        monthly.append({"month": m, "hours": len(observed),
                        "predicted_peak_q50_mw": pred_peak,
                        "observed_peak_mw": actual_peak,
                        "predicted_max_3h_ramp_mw": pred_max_ramp,
                        "observed_max_3h_ramp_mw": actual_max_ramp,
                        "reported_max_reserve_proxy_mw": pred_reserve,
                        "q95_exceedance_fraction": float(np.mean(observed > q95)),
                        "q05_lower_exceedance_fraction": float(np.mean(observed < q05))})
    shortfall = np.maximum(y - upper, 0.0)
    return {
        "test_hours": len(y), "months": len(monthly),
        "monthly_peak_q50_mae_mw": float(np.mean(peak_errors)),
        "monthly_max_3h_ramp_q50_mae_mw": float(np.mean(max_ramp_errors)),
        "hourly_ramp_q50_mae_mw": float(np.mean(hourly_ramp_errors)),
        "three_hour_ramp_q50_mae_mw": float(np.mean(three_hour_ramp_errors)),
        "q95_exceedance_fraction": float(np.mean(y > upper)),
        "q05_lower_exceedance_fraction": float(np.mean(y < lower)),
        "mean_positive_q95_exceedance_mw_per_hour": float(np.mean(shortfall)),
        "mean_q95_exceedance_mw_when_exceeded":
            float(np.mean(shortfall[shortfall > 0])) if np.any(shortfall > 0) else 0.0,
        "monthly": monthly,
    }


def evaluate(baseline: Path, native: Path, output: Path) -> dict:
    baseline, native, output = (p.resolve() for p in (baseline, native, output))
    actual_base = pd.read_csv(baseline / "actuals_2020.csv")
    actual_native = pd.read_csv(native / "actuals_2020.csv")
    if not actual_base.TIMESTAMP.equals(actual_native.TIMESTAMP) or \
            not np.allclose(actual_base.net_load_actual_mw,
                            actual_native.net_load_actual_mw):
        raise ValueError("baseline and native 2020 targets differ")
    results = {}
    for variant, root, prefix in (("baseline", baseline, ""),
                                  ("native_raw", native, "raw_"),
                                  ("native_calibrated", native, "calibrated_")):
        for method in ("copula", "independent"):
            forecast = pd.read_csv(root / f"net_load_{prefix}{method}_q01_q99_2020.csv")
            decisions = json.loads((root / f"decisions_{prefix}{method}_2020.json").read_text())
            results[f"{variant}_{method}"] = _variant(actual_base, forecast, decisions)
    report = {
        "experiment": "measured_2020_forecast_decision_proxy_audit",
        "nominal_q95_exceedance_fraction": 0.05,
        "no_operator_reserve_or_action_ground_truth": True,
        "interpretation": "Peaks, ramps and Q95 exceedance are measured forecast-risk proxies only; reserve activation, dispatch feasibility and operating cost cannot be scored from this dataset.",
        "variants": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.baseline, args.native, args.output)
    print(json.dumps({key: {metric: value for metric, value in result.items()
                            if metric != "monthly"}
                      for key, result in report["variants"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

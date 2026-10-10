"""Create a reproducible synchronized benchmark for joint fusion and decisions.

This is deliberately a *simulation* benchmark, not a replacement for measured
utility data.  It provides one common hourly grid, calibrated load/solar
marginals, correlated residuals, and labelled operating scenarios so the joint
Fusion and Decision Agents can be evaluated without silently pairing unrelated
GEFCom calendars.

Run from the repository root with the load environment, for example::

    agentic/.venv/bin/python -m fusion.synthetic_benchmark --output joint_outputs/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from .joint_pipeline import run_joint_pipeline


QUANTILES = np.arange(1, 100, dtype=float) / 100.0
QCOLS = [f"q{i:02d}" for i in range(1, 100)]


def _scenario_labels(n_days: int) -> np.ndarray:
    names = ("normal", "cloudy_low_solar", "high_demand", "solar_outage",
             "evening_ramp", "normal", "high_demand")
    return np.array([names[i % len(names)] for i in range(n_days) for _ in range(24)])


def _scenario_parameters(name: str) -> tuple[float, float, float]:
    """Return load multiplier, solar multiplier and cloud persistence."""
    return {
        "normal": (1.00, 1.00, 0.05),
        "cloudy_low_solar": (1.02, 0.35, 0.25),
        "high_demand": (1.18, 1.00, 0.05),
        "solar_outage": (1.00, 0.05, 0.35),
        "evening_ramp": (1.05, 0.85, 0.08),
    }.get(name, (1.00, 1.00, 0.05))


def create_benchmark(output: str | Path, *, start: str = "2020-01-01",
                     days: int = 14, capacity_mw: float = 60.0,
                     seed: int = 2026, calibration_days: int | None = None,
                     n_scenarios: int = 5000) -> dict:
    """Generate a temporal calibration/test split and evaluate only held-out hours."""
    if calibration_days is None:
        calibration_days = days // 2
    if days < 2 or not 1 <= calibration_days < days or capacity_mw <= 0:
        raise ValueError("require days >= 2, 1 <= calibration_days < days and positive capacity_mw")
    if n_scenarios < 2:
        raise ValueError("n_scenarios must be at least 2")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(start, periods=days * 24, freq="h")
    hour = stamps.hour.to_numpy() + stamps.minute.to_numpy() / 60.0
    labels = _scenario_labels(days)

    load_mult = np.array([_scenario_parameters(x)[0] for x in labels])
    solar_mult = np.array([_scenario_parameters(x)[1] for x in labels])
    # Smooth, interpretable grid-level profiles (MW).  The residual shocks are
    # correlated so dependence is estimable by the Fusion Agent.
    daily = 1.0 + 0.06 * np.sin(2 * np.pi * (stamps.dayofyear.to_numpy() / 7.0))
    morning = 18.0 * np.exp(-0.5 * ((hour - 8.0) / 2.2) ** 2)
    evening = 34.0 * np.exp(-0.5 * ((hour - 19.0) / 3.0) ** 2)
    base_load = (155.0 * daily + morning + evening) * load_mult
    solar_shape = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0)) ** 1.6
    base_solar_fraction = np.clip(0.92 * solar_shape * solar_mult, 0.0, 1.0)
    # Positive load shocks and reduced solar can occur together, as in a cloudy
    # high-demand period.  This is the controlled dependence used for testing.
    common = rng.normal(0.0, 1.0, len(stamps))
    load_shock = 5.5 * common + rng.normal(0.0, 2.0, len(stamps))
    solar_shock = -0.055 * common + rng.normal(0.0, 0.018, len(stamps))
    load_actual = np.maximum(1.0, base_load + load_shock)
    solar_actual_fraction = np.clip(base_solar_fraction + solar_shock, 0.0, 1.0)

    # Forecast centres contain modest, reproducible misspecification; the
    # quantile width changes by time of day and scenario, then is clipped at the
    # physical bounds.  These are forecast artifacts, not future observations.
    load_centre = np.maximum(1.0, base_load + 0.8 * np.sin(2 * np.pi * hour / 24.0))
    load_scale = 4.0 + 0.025 * load_centre
    solar_centre = np.clip(base_solar_fraction * 0.98, 0.0, 1.0)
    solar_scale = 0.025 + 0.06 * np.sqrt(np.maximum(solar_centre, 0.0))
    z = norm.ppf(QUANTILES)
    load_q = np.maximum(0.0, load_centre[:, None] + load_scale[:, None] * z[None, :])
    solar_q = np.clip(solar_centre[:, None] + solar_scale[:, None] * z[None, :], 0.0, 1.0)

    actuals = pd.DataFrame({
        "TIMESTAMP": stamps, "scenario": labels,
        "load_actual_MW": load_actual,
        "solar_actual_fraction": solar_actual_fraction,
        "solar_actual_MW": solar_actual_fraction * capacity_mw,
        "net_load_actual_MW": load_actual - solar_actual_fraction * capacity_mw,
    })
    load_frame = pd.DataFrame(
        {"TIMESTAMP": stamps, "actual_LOAD": load_actual,
         **{col: load_q[:, i] for i, col in enumerate(QCOLS)}}
    )
    solar_frame = pd.DataFrame(
        {"TIMESTAMP": stamps,
         **{col: solar_q[:, i] for i, col in enumerate(QCOLS)}}
    )
    index = pd.DataFrame({"TIMESTAMP": stamps, "ZONEID": 1,
                          "POWER": solar_actual_fraction, "scenario": labels})
    actuals.to_csv(root / "synthetic_actuals.csv", index=False)
    load_frame.to_csv(root / "load_predictions.csv", index=False)
    solar_frame.to_csv(root / "solar_predictions.csv", index=False)
    index.to_csv(root / "solar_index.csv", index=False)

    split = calibration_days * 24
    load_residual = load_actual[:split] - load_q[:split, 49]
    solar_residual = solar_actual_fraction[:split] - solar_q[:split, 49]
    dependence = float(spearmanr(load_residual, solar_residual).statistic)
    if not np.isfinite(dependence):
        raise ValueError("calibration residual dependence is not finite")
    # Evaluation artifacts contain only future hours.  The explicit
    # calibration-period coefficient takes precedence over any actuals attached
    # by the adapters, so no held-out observation is used during fusion.
    load_frame.iloc[split:].to_csv(root / "load_predictions_test.csv", index=False)
    solar_frame.iloc[split:].to_csv(root / "solar_predictions_test.csv", index=False)
    index.iloc[split:].to_csv(root / "solar_index_test.csv", index=False)

    report = run_joint_pipeline(
        root / "load_predictions_test.csv", root / "solar_predictions_test.csv",
        root / "solar_index_test.csv", root / "joint_run",
        solar_capacity_mw=capacity_mw, solar_location_value=1,
        n_scenarios=n_scenarios, seed=seed, dependence=dependence,
        solar_capacity_basis="ASSUMED: installed capacity of simulated grid",
    )
    forecast = pd.read_csv(report["forecast_path"])
    evaluation_actuals = actuals.iloc[split:].reset_index(drop=True)
    if not pd.DatetimeIndex(forecast["TIMESTAMP"]).equals(
            pd.DatetimeIndex(evaluation_actuals["TIMESTAMP"])):
        raise ValueError("forecast and held-out observation timestamps differ")
    actual = evaluation_actuals["net_load_actual_MW"].to_numpy()
    pred = forecast[QCOLS].to_numpy(float)
    pinball = np.mean(np.maximum(QUANTILES[None, :] * (actual[:, None] - pred),
                                 (QUANTILES[None, :] - 1.0) * (actual[:, None] - pred)))
    q05, q95 = forecast["q05"].to_numpy(float), forecast["q95"].to_numpy(float)
    evaluation = {
        "benchmark_type": "controlled_synthetic_synchronized_grid_benchmark",
        "not_measured_utility_data": True,
        "n_timestamps": len(forecast), "days": days,
        "calibration_days": calibration_days,
        "evaluation_days": days - calibration_days,
        "calibration_start": str(stamps[0]),
        "calibration_end": str(stamps[split - 1]),
        "evaluation_start": str(stamps[split]),
        "evaluation_end": str(stamps[-1]),
        "calibration_rank_correlation": dependence,
        "dependence_uses_evaluation_actuals": False,
        "solar_capacity_mw": capacity_mw, "seed": seed,
        "n_scenarios": n_scenarios,
        "scenarios": sorted(set(labels.tolist())),
        "mean_pinball_loss_mw": float(pinball),
        "coverage_q05_q95": float(np.mean((actual >= q05) & (actual <= q95))),
        "mean_interval_width_q05_q95_mw": float(np.mean(q95 - q05)),
        "quantile_crossing_count": int(np.sum(np.diff(pred, axis=1) < -1e-10)),
        "actual_net_load_min_mw": float(actual.min()),
        "actual_net_load_max_mw": float(actual.max()),
        "fusion_forecast": report["forecast_path"],
        "decision_report": report["decision_report_path"],
    }
    (root / "joint_evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "benchmark_type": evaluation["benchmark_type"],
        "purpose": "Synchronized proof-of-concept for Fusion and Decision Agents",
        "seed": seed, "start": str(stamps[0]), "end": str(stamps[-1]),
        "capacity_mw": capacity_mw, "n_scenarios": n_scenarios,
        "calibration_days": calibration_days,
        "evaluation_days": days - calibration_days,
        "calibration_rank_correlation": dependence,
        "scenario_definitions": {
            "normal": "baseline daily load and solar",
            "cloudy_low_solar": "reduced PV availability with slightly higher demand",
            "high_demand": "18 percent demand multiplier",
            "solar_outage": "near-zero PV availability",
            "evening_ramp": "elevated evening demand and reduced PV",
        },
        "limitations": [
            "Synthetic observations are not a claim of measured grid behaviour.",
            "Forecast marginals are synthetic assumptions; dependence is estimated from earlier synthetic calibration hours only.",
            "Measured synchronized utility data remain future validation work.",
        ],
        "inputs": ["synthetic_actuals.csv", "load_predictions.csv",
                   "solar_predictions.csv", "solar_index.csv",
                   "load_predictions_test.csv", "solar_predictions_test.csv",
                   "solar_index_test.csv"],
        "outputs": ["joint_run/net_load_forecast.csv", "joint_run/decision_report.json",
                    "joint_run/joint_manifest.json", "joint_evaluation.json"],
    }
    (root / "synthetic_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"root": str(root), "evaluation": evaluation, "report": report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("joint_outputs/synthetic_benchmark"))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--capacity-mw", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-days", type=int, default=None)
    parser.add_argument("--scenarios", type=int, default=5000)
    args = parser.parse_args(argv)
    result = create_benchmark(args.output, start=args.start, days=args.days,
                               capacity_mw=args.capacity_mw, seed=args.seed,
                               calibration_days=args.calibration_days,
                               n_scenarios=args.scenarios)
    print(json.dumps(result["evaluation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

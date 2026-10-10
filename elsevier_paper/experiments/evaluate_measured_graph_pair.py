"""Fuse and evaluate two completed measured forecasting graph artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fusion.decision_agent import make_decisions
from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS
from fusion.fusion_agent import fuse_forecasts

from .run_opsd_baseline import _score


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_graph(path: Path, domain: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report = json.loads((path / "graph_evaluation.json").read_text())
    if (report.get("experiment"), report.get("domain"), report.get("area"),
            report.get("horizon"), report.get("unit"), report.get("test_hours")) != (
            "adapted_eight_node_measured_forecast_graph", domain, "DE", "24h", "MW", 8760):
        raise ValueError(f"{domain} graph report does not match the joint protocol")
    if _sha(path / "test_q01_q99.csv") != report["forecast_sha256"]:
        raise ValueError(f"{domain} forecast artifact changed after graph evaluation")
    forecast = pd.read_csv(path / "test_q01_q99.csv")
    actual = pd.read_csv(path / "test_actuals.csv")
    residual = pd.read_csv(path / "calibration_residuals.csv")
    if not forecast.TIMESTAMP.equals(actual.TIMESTAMP):
        raise ValueError(f"{domain} graph forecast and actual times differ")
    if list(forecast.columns) != ["TIMESTAMP", *QUANTILE_COLUMNS]:
        raise ValueError(f"{domain} graph forecast does not have Q01--Q99")
    values = forecast[list(QUANTILE_COLUMNS)].to_numpy()
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < -1e-9).any():
        raise ValueError(f"{domain} graph forecast is nonfinite or crosses quantiles")
    return report, forecast, actual, residual


def run(load_dir: Path, solar_dir: Path, inputs: Path, output: Path, *,
        scenarios: int = 1000, seed: int = 2026) -> dict:
    if scenarios < 100:
        raise ValueError("at least 100 scenarios are required")
    load_dir, solar_dir, inputs, output = map(Path.resolve, (load_dir, solar_dir, inputs, output))
    load_report, load, load_actual, load_cal = _read_graph(load_dir, "load")
    solar_report, solar, solar_actual, solar_cal = _read_graph(solar_dir, "solar")
    if load_report["source_input_manifest_sha256"] != solar_report["source_input_manifest_sha256"]:
        raise ValueError("forecast graphs used different source manifests")
    if not load.TIMESTAMP.equals(solar.TIMESTAMP):
        raise ValueError("load and solar test timestamps differ")
    if not load_cal.TIMESTAMP.equals(solar_cal.TIMESTAMP):
        raise ValueError("load and solar calibration timestamps differ")
    stamp = pd.to_datetime(load.TIMESTAMP, utc=True)
    if len(stamp) != 8760 or stamp.duplicated().any() or not stamp.is_monotonic_increasing:
        raise ValueError("2019 test timestamps are incomplete or unordered")
    calibration_time = pd.to_datetime(load_cal.TIMESTAMP, utc=True)
    dep = calibration_time >= pd.Timestamp("2018-10-01T00:00:00Z")
    if dep.sum() < 1000:
        raise ValueError("dependence calibration block is incomplete")
    rho = float(spearmanr((load_cal.actual_mw - load_cal.q50_mw)[dep],
                          (solar_cal.actual_mw - solar_cal.q50_mw)[dep]).statistic)
    if not np.isfinite(rho):
        raise ValueError("dependence calibration failed")
    rho = float(np.clip(rho, -0.99, 0.99))
    capacity = max(pd.read_csv(inputs / f"solar_{split}.csv", usecols=["capacity_lag25"])
                   .capacity_lag25.max() for split in ("train", "calibration"))
    load_bundle = ForecastBundle("measured adapted load graph", load, "MW", "DE", "24h",
                                 physical_min=0.0)
    solar_bundle = ForecastBundle("measured adapted solar graph", solar, "MW", "DE", "24h",
                                  metadata={"capacity_mw": float(capacity)},
                                  physical_min=0.0, physical_max=float(capacity))
    actual_net = load_actual.actual_mw.to_numpy() - solar_actual.actual_mw.to_numpy()
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    month = stamp.dt.tz_convert("Europe/Berlin").dt.month.to_numpy()
    for method, dependence in (("copula", rho), ("independent", 0.0)):
        forecasts, decisions = [], []
        for number in range(1, 13):
            take = month == number
            fused = fuse_forecasts(
                load_bundle.replace_frame(load.loc[take].reset_index(drop=True)),
                solar_bundle.replace_frame(solar.loc[take].reset_index(drop=True)),
                n_scenarios=scenarios, seed=seed + number,
                dependence=dependence, output_location="DE")
            decision = make_decisions(fused)
            forecasts.append(fused.forecast)
            decisions.append({"month": number, "summary": decision.summary,
                              "risk_flags": decision.risk_flags,
                              "recommendations": decision.recommendations})
        frame = pd.concat(forecasts, ignore_index=True)
        if not pd.to_datetime(frame.TIMESTAMP, utc=True).reset_index(drop=True).equals(
                stamp.reset_index(drop=True)):
            raise ValueError("monthly fusion changed forecast timestamps")
        forecast_path = output / f"net_load_{method}_q01_q99_2019.csv"
        frame.to_csv(forecast_path, index=False)
        (output / f"decisions_{method}_2019.json").write_text(
            json.dumps(decisions, indent=2) + "\n")
        results[method] = {"metrics": _score(actual_net, frame[list(QUANTILE_COLUMNS)].to_numpy()),
                           "forecast_sha256": _sha(forecast_path)}
    report = {"experiment": "adapted_measured_graph_pair_fusion_decision",
              "area": "DE", "horizon": "24h", "unit": "MW", "test_hours": len(stamp),
              "scenarios_per_hour": scenarios, "seed": seed,
              "dependence_rank_correlation": rho,
              "load_graph_report_sha256": _sha(load_dir / "graph_evaluation.json"),
              "solar_graph_report_sha256": _sha(solar_dir / "graph_evaluation.json"),
              "net_load": results,
              "caveat": "Adapted eight-node forecast graphs with scripted review; decision outputs are proxies, not operator-action validation."}
    (output / "joint_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("load_dir", type=Path)
    parser.add_argument("solar_dir", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run(args.load_dir, args.solar_dir, args.inputs, args.output,
                         scenarios=args.scenarios, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

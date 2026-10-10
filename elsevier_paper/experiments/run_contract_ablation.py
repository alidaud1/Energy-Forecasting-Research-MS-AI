"""Controlled hand-off fault injection, not an agentic accuracy comparison.

Compare the existing ForecastBundle/fuse_forecasts path with explicitly
unchecked positional fusion using identical random draws and numerical rules.
Archived measured marginal forecasts are unchanged. No LLM is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS
from fusion.fusion_agent import fuse_forecasts

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
Q = list(QUANTILE_COLUMNS)
LEVELS = np.arange(1, 100) / 100.0


def unchecked_positional_fusion(load: pd.DataFrame, solar: pd.DataFrame, *,
                                capacity: float, scenarios: int = 1000,
                                seed: int = 2026, rank_correlation: float = -.25) -> np.ndarray:
    """An explicit counterfactual that assumes aligned, finite MW quantiles.

    It ignores metadata and timestamps; it is not the original pipelines,
    nor a claim that all conventional pipelines lack validation.
    """
    left, right = load[Q].to_numpy(float), solar[Q].to_numpy(float)
    z1, z2 = np.random.default_rng(seed).standard_normal((2, scenarios, len(load)))
    rho = np.clip(2 * np.sin(np.pi * rank_correlation / 6), -.999, .999)
    uniforms = ndtr(z1), ndtr(rho * z1 + np.sqrt(1 - rho * rho) * z2)

    def sample(values, u):
        indices = np.clip(np.searchsorted(LEVELS, u, side="left"), 1, 98)
        low, high = LEVELS[indices - 1], LEVELS[indices]
        a = np.take_along_axis(values.T, indices - 1, axis=0)
        b = np.take_along_axis(values.T, indices, axis=0)
        return a + (b - a) * (u - low) / (high - low)

    net = np.clip(sample(left, uniforms[0]), 0, None) - np.clip(sample(right, uniforms[1]), 0, capacity)
    return np.quantile(net, LEVELS, axis=0).T


def checked_fusion(load, solar, *, capacity, solar_unit="MW", solar_horizon="24h",
                   solar_location="DE", **settings):
    left = ForecastBundle("archived load graph", load, "MW", "DE", "24h", physical_min=0)
    scale = 1000 if solar_unit == "kW" else 1
    right = ForecastBundle("archived solar graph", solar, solar_unit, solar_location,
                           solar_horizon, metadata={"capacity_mw": capacity},
                           physical_min=0, physical_max=capacity * scale)
    return fuse_forecasts(left, right, n_scenarios=settings.get("scenarios", 1000),
                          seed=settings.get("seed", 2026),
                          dependence=settings.get("rank_correlation", -.25), output_location="DE").forecast[Q].to_numpy()


def execute_case(name, load, solar, *, capacity, metadata=None):
    row = {"case": name}
    for label, function in (("checked", checked_fusion), ("unchecked", unchecked_positional_fusion)):
        try:
            settings = (metadata or {}) if label == "checked" else {}
            values = function(load, solar, capacity=capacity, **settings)
            row[label] = "finite_output" if np.isfinite(values).all() else "nonfinite_output"
        except (ValueError, KeyError, TypeError) as error:
            row[label] = "rejected"
            row[label + "_error"] = str(error)
    return row


def run(output: Path | None = None) -> dict:
    graph = BASE / "opsd_measured_full_graph"
    load_path, solar_path = graph / "load/test_q01_q99.csv", graph / "solar/test_q01_q99.csv"
    load, solar = pd.read_csv(load_path).iloc[:72].copy(), pd.read_csv(solar_path).iloc[:72].copy()
    if not load.TIMESTAMP.equals(solar.TIMESTAMP):
        raise ValueError("clean measured artifacts are not aligned")
    capacity = float(pd.read_csv(BASE / "opsd_agent_port/solar_calibration.csv",
                                usecols=["capacity_lag25"]).capacity_lag25.max())
    clean_checked = checked_fusion(load, solar, capacity=capacity)
    clean_unchecked = unchecked_positional_fusion(load, solar, capacity=capacity)
    np.testing.assert_allclose(clean_checked, clean_unchecked, rtol=0, atol=1e-9)
    cases = []

    def add(name, frame, metadata=None):
        cases.append(execute_case(name, load, frame, capacity=capacity, metadata=metadata))

    add("missing_quantile", solar.drop(columns="q37"))
    frame = solar.copy(); frame.loc[0, "q50"] = np.nan
    add("nonfinite_quantile", frame)
    frame = solar.copy(); frame.loc[0, "q01"] = frame.loc[0, "q99"] + 1
    add("crossing_quantiles", frame)
    frame = solar.copy(); frame.loc[1, "TIMESTAMP"] = frame.loc[0, "TIMESTAMP"]
    add("duplicate_timestamp", frame)
    add("reversed_timestamps", solar.iloc[::-1].reset_index(drop=True))
    frame = solar.copy(); frame.loc[0, "TIMESTAMP"] = "not-a-date"
    add("invalid_timestamp", frame)
    frame = solar.copy(); frame.TIMESTAMP = (pd.to_datetime(frame.TIMESTAMP) + pd.Timedelta(days=400)).astype(str)
    add("no_timestamp_overlap", frame)
    add("horizon_mismatch", solar, {"solar_horizon": "48h"})
    add("unsupported_unit", solar, {"solar_unit": "MWh"})
    frame = solar.copy(); frame[Q] *= 1000
    checked = checked_fusion(load, frame, capacity=capacity, solar_unit="kW")
    unchecked = unchecked_positional_fusion(load, frame, capacity=capacity)
    np.testing.assert_allclose(clean_checked, checked, rtol=0, atol=1e-9)
    conversion = {"case": "declared_kW_conversion", "checked": "correct_unit_conversion",
                  "unchecked": "finite_output_wrong_scale",
                  "checked_max_abs_difference_mw": float(np.max(np.abs(checked - clean_checked))),
                  "unchecked_max_abs_difference_mw": float(np.max(np.abs(unchecked - clean_checked)))}
    # A deliberate negative control: geographic labels are recorded, not
    # physically authenticated. Do not turn this experiment into a safety claim.
    negative = execute_case("unverified_geographic_label", load, solar, capacity=capacity,
                            metadata={"solar_location": "different_grid"})
    report = {"experiment": "forecast_contract_vs_unchecked_positional_fusion",
              "n_hours": 72, "scenarios_per_hour": 1000, "seed": 2026,
              "rank_correlation": -.25, "dependence_note": "fixed test setting, not an estimated forecast-skill claim",
              "source_sha256": {"load": hashlib.sha256(load_path.read_bytes()).hexdigest(),
                                "solar": hashlib.sha256(solar_path.read_bytes()).hexdigest()},
              "clean_max_abs_difference_mw": float(np.max(np.abs(clean_checked - clean_unchecked))),
              "fault_cases": cases, "declared_unit_case": conversion, "negative_control": negative,
              "summary": {"malformed_cases": len(cases),
                          "checked_rejections": sum(c["checked"] == "rejected" for c in cases),
                          "unchecked_rejections": sum(c["unchecked"] == "rejected" for c in cases),
                          "unchecked_finite_outputs": sum(c["unchecked"] == "finite_output" for c in cases),
                          "unchecked_nonfinite_outputs": sum(c["unchecked"] == "nonfinite_output" for c in cases)},
              "scope": "controlled single-fault engineering check; not fault prevalence, deployment safety, runtime benefit or agentic forecasting improvement"}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "contract_ablation.json").write_text(json.dumps(report, indent=2) + "\n")
        pd.DataFrame([*cases, conversion, negative]).to_csv(output / "contract_ablation.csv", index=False)
    return report


if __name__ == "__main__":
    result = run(BASE / "contract_ablation")
    print(json.dumps({"summary": result["summary"],
                      "clean_max_abs_difference_mw": result["clean_max_abs_difference_mw"],
                      "unit_case": result["declared_unit_case"],
                      "negative_control": result["negative_control"]}, indent=2))

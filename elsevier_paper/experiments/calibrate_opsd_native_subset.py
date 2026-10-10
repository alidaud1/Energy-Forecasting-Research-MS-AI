"""Evaluate frozen native quantile models on a new 2020 measured holdout.

Quantile-offset recalibration uses October–November 2018 only. December 2018
is reserved for the load–solar dependence coefficient. Neither the 2019
observations nor the 2020 outcomes enter either calibration step. The 2020
capacity used by the model is explicitly a last-published-2019 proxy.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fusion.decision_agent import make_decisions
from fusion.forecast_contract import ForecastBundle
from fusion.fusion_agent import fuse_forecasts

from .run_opsd_baseline import LEVELS, _forecast_frame, _score
from .run_opsd_native_subset import (QUANTILES, _clean_quantiles, _matrix,
                                     _native_trainers, _sha)


def quantile_offsets(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Empirical additive correction for each marginal quantile (no guarantee)."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(actual), 99) or len(actual) < 100 or \
            not np.isfinite(prediction).all() or not np.isfinite(actual).all():
        raise ValueError("invalid marginal calibration arrays")
    return np.array([np.quantile(actual - prediction[:, i], q)
                     for i, q in enumerate(LEVELS)])


def _predict(models: list, features: np.ndarray, *, upper: float = np.inf) -> np.ndarray:
    if len(models) != 99:
        raise ValueError("expected exactly 99 native quantile estimators")
    return _clean_quantiles(np.column_stack([model.predict(features) for model in models]),
                            upper=upper)


def _rho(load_y: np.ndarray, solar_y: np.ndarray,
         load_q: np.ndarray, solar_q_mw: np.ndarray) -> float:
    value = float(spearmanr(load_y - load_q[:, 49],
                            solar_y - solar_q_mw[:, 49]).statistic)
    if not np.isfinite(value):
        raise ValueError("undefined dependence coefficient")
    return float(np.clip(value, -0.99, 0.99))


def _fuse(load: pd.DataFrame, solar: pd.DataFrame, actual_net: np.ndarray,
          rho: float, bound: float, output: Path, label: str,
          *, scenarios: int, seed: int) -> dict:
    load_bundle = ForecastBundle("OPSD-DE frozen native load GBR", load, "MW", "DE", "24h",
                                 physical_min=0.0)
    solar_bundle = ForecastBundle("OPSD-DE frozen native solar XGBoost", solar, "MW", "DE", "24h",
                                  metadata={"capacity_mw": bound},
                                  physical_min=0.0, physical_max=bound)
    stamp = pd.to_datetime(load.TIMESTAMP, utc=True)
    months = stamp.dt.tz_convert("Europe/Berlin").dt.month.to_numpy()
    scores = {}
    for method, dependence in (("copula", rho), ("independent", 0.0)):
        frames, decisions = [], []
        for month in range(1, 7):
            take = months == month
            fused = fuse_forecasts(
                load_bundle.replace_frame(load.loc[take].reset_index(drop=True)),
                solar_bundle.replace_frame(solar.loc[take].reset_index(drop=True)),
                n_scenarios=scenarios, seed=seed + month,
                dependence=dependence, output_location="DE")
            action = make_decisions(fused)
            frames.append(fused.forecast)
            decisions.append({"month": month, "summary": action.summary,
                              "risk_flags": action.risk_flags,
                              "recommendations": action.recommendations})
        merged = pd.concat(frames, ignore_index=True)
        if not pd.to_datetime(merged.TIMESTAMP).equals(pd.to_datetime(load.TIMESTAMP)):
            raise ValueError("monthly fusion broke timestamp alignment")
        merged.to_csv(output / f"net_load_{label}_{method}_q01_q99_2020.csv", index=False)
        (output / f"decisions_{label}_{method}_2020.json").write_text(
            json.dumps(decisions, indent=2) + "\n")
        scores[method] = _score(actual_net, merged[QUANTILES].to_numpy(float))
    return scores


def run(inputs: Path, native: Path, holdout: Path, output: Path,
        *, scenarios: int = 1000, seed: int = 2026) -> dict:
    inputs, native, holdout, output = (p.resolve() for p in
                                       (inputs, native, holdout, output))
    if scenarios < 100:
        raise ValueError("at least 100 fusion scenarios are required")
    manifest = json.loads((inputs / "input_manifest.json").read_text())
    held = json.loads((holdout / "manifest.json").read_text())
    if held["test_hours"] != 4367 or held["capacity_proxy_mw"] <= 0:
        raise ValueError("invalid 2020 holdout manifest")
    load_cal = pd.read_csv(inputs / "load_calibration.csv")
    solar_cal = pd.read_csv(inputs / "solar_calibration.csv")
    load_test = pd.read_csv(holdout / "load_test_2020.csv")
    solar_test = pd.read_csv(holdout / "solar_test_2020.csv")
    actual = pd.read_csv(holdout / "actuals_2020.csv")
    for a, b in ((load_cal, solar_cal), (load_test, solar_test), (load_test, actual)):
        if not a.TIMESTAMP.equals(b.TIMESTAMP):
            raise ValueError("load, solar or actual timestamp mismatch")
    stamp = pd.to_datetime(load_cal.TIMESTAMP, utc=True)
    offset_period = ((stamp >= pd.Timestamp("2018-10-01T00:00:00Z")) &
                     (stamp < pd.Timestamp("2018-12-01T00:00:00Z"))).to_numpy()
    dependence_period = (stamp >= pd.Timestamp("2018-12-01T00:00:00Z")).to_numpy()
    if offset_period.sum() < 1400 or dependence_period.sum() < 700 or \
            (offset_period & dependence_period).any():
        raise ValueError("2018 calibration blocks are missing or overlapping")
    if stamp.iloc[-1] >= pd.Timestamp("2019-01-01T00:00:00Z"):
        raise ValueError("calibration must precede the published 2019 results")
    # Import model classes so their frozen, isolated checkpoints can unpickle.
    _native_trainers()
    with (native / "checkpoints" / "Load_GBR_OPSD_DE.pkl").open("rb") as handle:
        load_models = pickle.load(handle)
    with (native / "checkpoints" / "XGBoost_best.pkl").open("rb") as handle:
        solar_models = pickle.load(handle)
    l_cols, s_cols = manifest["load_features"], manifest["solar_features"]
    l_cal = _predict(load_models, _matrix(load_cal, l_cols))
    l_test = _predict(load_models, _matrix(load_test, l_cols))
    s_cal = _predict(solar_models, _matrix(solar_cal, s_cols), upper=1.0)
    s_test = _predict(solar_models, _matrix(solar_test, s_cols), upper=1.0)
    l_delta = quantile_offsets(load_cal.LOAD.to_numpy()[offset_period],
                               l_cal[offset_period])
    s_delta = quantile_offsets(solar_cal.POWER.to_numpy()[offset_period],
                               s_cal[offset_period])
    l_cal_adjusted = _clean_quantiles(l_cal + l_delta)
    s_cal_adjusted = _clean_quantiles(s_cal + s_delta, upper=1.0)
    l_test_adjusted = _clean_quantiles(l_test + l_delta)
    s_test_adjusted = _clean_quantiles(s_test + s_delta, upper=1.0)
    dep_cap = solar_cal.capacity_lag25.to_numpy()[dependence_period]
    dep_load = load_cal.LOAD.to_numpy()[dependence_period]
    dep_solar = solar_cal.POWER.to_numpy()[dependence_period] * dep_cap
    raw_rho = _rho(dep_load, dep_solar, l_cal[dependence_period],
                   s_cal[dependence_period] * dep_cap[:, None])
    calibrated_rho = _rho(dep_load, dep_solar, l_cal_adjusted[dependence_period],
                          s_cal_adjusted[dependence_period] * dep_cap[:, None])
    test_cap = solar_test.capacity_lag25.to_numpy()
    bound = float(held["capacity_proxy_mw"])
    output.mkdir(parents=True, exist_ok=True)
    actual.to_csv(output / "actuals_2020.csv", index=False)
    test_stamp = pd.to_datetime(load_test.TIMESTAMP, utc=True)
    actual_l = actual.load_actual_mw.to_numpy()
    actual_s = actual.solar_actual_mw.to_numpy()
    actual_net = actual.net_load_actual_mw.to_numpy()
    variants = {}
    for label, l_q, s_q, rho in (("raw", l_test, s_test, raw_rho),
                                 ("calibrated", l_test_adjusted,
                                  s_test_adjusted, calibrated_rho)):
        solar_mw = s_q * test_cap[:, None]
        l_frame = _forecast_frame(test_stamp, l_q)
        s_frame = _forecast_frame(test_stamp, solar_mw)
        l_frame.to_csv(output / f"load_{label}_q01_q99_2020.csv", index=False)
        s_frame.to_csv(output / f"solar_{label}_q01_q99_2020.csv", index=False)
        net = _fuse(l_frame, s_frame, actual_net, rho, bound, output, label,
                    scenarios=scenarios, seed=seed)
        variants[label] = {"rank_correlation_dec_2018": rho,
                           "load": _score(actual_l, l_q),
                           "solar": _score(actual_s, solar_mw),
                           "net_load": net}
    report = {
        "experiment": "OPSD_DE_frozen_native_GBR_XGBoost_2020H1_holdout",
        "not_full_LangGraph_rerun": True,
        "source_input_manifest_sha256": _sha(inputs / "input_manifest.json"),
        "source_2020_manifest_sha256": _sha(holdout / "manifest.json"),
        "checkpoint_sha256": {
            "load": _sha(native / "checkpoints" / "Load_GBR_OPSD_DE.pkl"),
            "solar": _sha(native / "checkpoints" / "XGBoost_best.pkl")},
        "calibration_method": "2018-Oct-Nov empirical additive quantile offsets; monotone rearrangement and physical clipping; exploratory method chosen after review of 2019 scores, assessed on new 2020 holdout; no formal coverage guarantee",
        "offset_calibration_hours": int(offset_period.sum()),
        "dependence_calibration_hours": int(dependence_period.sum()),
        "test_hours": len(actual), "scenarios_per_hour": scenarios, "seed": seed,
        "solar_capacity_caveat": held["capacity_status"],
        "variants": variants,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = run(args.inputs, args.native, args.holdout, args.output,
                 scenarios=args.scenarios, seed=args.seed)
    print(json.dumps({"test_hours": result["test_hours"],
                      "offset_calibration_hours": result["offset_calibration_hours"],
                      "variants": result["variants"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

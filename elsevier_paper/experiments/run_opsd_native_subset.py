"""Isolated measured-data port of load GBR and solar XGBoost trainers.

Uses the original quantile-training functions, not the original GEFCom
preprocessing, date splits, agent graph or output directories. The first
half of 2018-H2 is model validation, the second half is dependence
calibration, and 2019 is untouched test. This is a two-model architecture
port, not a full multi-agent retraining claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fusion.decision_agent import make_decisions
from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS
from fusion.fusion_agent import fuse_forecasts

from .run_opsd_baseline import _score, _forecast_frame

ROOT = Path(__file__).resolve().parents[2]
QUANTILES = list(QUANTILE_COLUMNS)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _native_trainers():
    # Load project uses top-level imports; solar uses package-relative imports.
    # Do not import either graph/main entry point, which would change their
    # configured output state. The functions called below are model-only.
    load_path = str(ROOT / "agentic" / "pipeline")
    solar_path = str(ROOT / "agentic_solar" / "solar-forecasting-GEFCom2014")
    if load_path not in sys.path:
        sys.path.insert(0, load_path)
    load = importlib.import_module("model_training")
    if solar_path not in sys.path:
        sys.path.insert(0, solar_path)
    solar = importlib.import_module("pipeline.model_training")
    return load, solar


def _matrix(table: pd.DataFrame, names: list[str]) -> np.ndarray:
    x = table[names].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("model feature matrix contains nonfinite values")
    return x


def _clean_quantiles(array: np.ndarray, *, upper: float | np.ndarray = np.inf) -> np.ndarray:
    result = np.asarray(array, dtype=float)
    if result.ndim != 2 or result.shape[1] != 99 or not np.isfinite(result).all():
        raise ValueError("native trainer must return 99 finite quantiles per hour")
    result = np.maximum.accumulate(result, axis=1)
    result = np.clip(result, 0.0, upper)
    if (np.diff(result, axis=1) < -1e-9).any():
        raise ValueError("postprocessed quantiles still cross")
    return result


def run(inputs: Path, output: Path, *, scenarios: int = 1000,
        seed: int = 2026, load_estimators: int = 50,
        solar_estimators: int = 80) -> dict:
    inputs, output = inputs.resolve(), output.resolve()
    manifest = json.loads((inputs / "input_manifest.json").read_text())
    if manifest.get("model_training_completed") is not False or manifest.get("area") != "DE":
        raise ValueError("expected the verified but untrained OPSD-DE input hand-off")
    tables = {(domain, split): pd.read_csv(inputs / f"{domain}_{split}.csv")
              for domain in ("load", "solar")
              for split in ("train", "calibration", "test")}
    for split in ("train", "calibration", "test"):
        if not tables["load", split].TIMESTAMP.equals(tables["solar", split].TIMESTAMP):
            raise ValueError(f"{split}: load and solar timestamps differ")
    if scenarios < 100 or solar_estimators < 2 or load_estimators < 2:
        raise ValueError("insufficient scenario or tree count")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    load_native, solar_native = _native_trainers()
    load_native.GBR_N_ESTIMATORS = load_estimators
    solar_native.MODEL_DIR = str(checkpoint_dir)
    # Set the RNG after importing each native module (both initialize a seed).
    np.random.seed(seed)

    train_load = tables["load", "train"]
    train_solar = tables["solar", "train"]
    cal_load = tables["load", "calibration"]
    cal_solar = tables["solar", "calibration"]
    test_load = tables["load", "test"]
    test_solar = tables["solar", "test"]
    cal_times = pd.to_datetime(cal_load.TIMESTAMP, utc=True)
    # Validation for the native solar trainer's early stopping is disjoint
    # from residual-dependence calibration. Both precede the 2019 test year.
    validation = (cal_times < pd.Timestamp("2018-10-01T00:00:00Z")).to_numpy()
    dependence = ~validation
    if min(validation.sum(), dependence.sum()) < 1000:
        raise ValueError("validation/dependence subperiods are too short")
    load_cols = manifest["load_features"]
    solar_cols = manifest["solar_features"]

    load_models, load_cal_raw, load_test_raw = load_native.train_gbr(
        _matrix(train_load, load_cols), train_load.LOAD.to_numpy(np.float32),
        _matrix(cal_load, load_cols), cal_load.LOAD.to_numpy(np.float32),
        _matrix(test_load, load_cols))
    with (checkpoint_dir / "Load_GBR_OPSD_DE.pkl").open("wb") as handle:
        pickle.dump(load_models, handle)
    solar_test_raw, solar_early_raw = solar_native.train_xgboost(
        _matrix(train_solar, solar_cols), train_solar.POWER.to_numpy(np.float32),
        _matrix(cal_solar.loc[validation], solar_cols),
        cal_solar.loc[validation, "POWER"].to_numpy(np.float32),
        _matrix(test_solar, solar_cols),
        hpo_params={"n_estimators": solar_estimators, "max_depth": 4})
    # The original solar trainer returns only its early-stop validation and
    # test forecasts. Obtain later, disjoint dependence-period predictions
    # from its isolated saved list of native quantile estimators.
    with (checkpoint_dir / "XGBoost_best.pkl").open("rb") as handle:
        solar_models = pickle.load(handle)
    solar_dep_raw = np.column_stack([
        model.predict(_matrix(cal_solar.loc[dependence], solar_cols))
        for model in solar_models
    ])
    del solar_models

    load_cal = _clean_quantiles(load_cal_raw)
    load_test = _clean_quantiles(load_test_raw)
    _clean_quantiles(solar_early_raw, upper=1.0)
    solar_dep = _clean_quantiles(solar_dep_raw, upper=1.0)
    solar_test = _clean_quantiles(solar_test_raw, upper=1.0)
    # Native solar predictions are capacity fractions. Restore MW with the
    # capacity that was already observed 25 hours before each target interval.
    dep_cap = cal_solar.loc[dependence, "capacity_lag25"].to_numpy()
    test_cap = test_solar.capacity_lag25.to_numpy()
    solar_dep_mw = solar_dep * dep_cap[:, None]
    solar_test_mw = solar_test * test_cap[:, None]
    load_dep_actual = cal_load.loc[dependence, "LOAD"].to_numpy()
    solar_dep_actual = cal_solar.loc[dependence, "POWER"].to_numpy() * dep_cap
    rank_corr = float(spearmanr(load_dep_actual - load_cal[dependence, 49],
                                solar_dep_actual - solar_dep_mw[:, 49]).statistic)
    if not np.isfinite(rank_corr):
        raise ValueError("dependence calibration failed")
    rank_corr = float(np.clip(rank_corr, -0.99, 0.99))

    stamp = pd.to_datetime(test_load.TIMESTAMP, utc=True)
    load_forecast = _forecast_frame(stamp, load_test)
    solar_forecast = _forecast_frame(stamp, solar_test_mw)
    load_forecast.to_csv(output / "load_GBR_q01_q99_2019.csv", index=False)
    solar_forecast.to_csv(output / "solar_XGBoost_q01_q99_2019.csv", index=False)
    actual_load = test_load.LOAD.to_numpy()
    actual_solar = test_solar.POWER.to_numpy() * test_cap
    actual_net = actual_load - actual_solar
    pd.DataFrame({"TIMESTAMP": load_forecast.TIMESTAMP,
                  "load_actual_mw": actual_load,
                  "solar_actual_mw": actual_solar,
                  "net_load_actual_mw": actual_net}).to_csv(output / "actuals_2019.csv", index=False)
    bound = float(max(train_solar.capacity_lag25.max(), cal_solar.capacity_lag25.max()))
    load_bundle = ForecastBundle("native load GBR adapted to OPSD", load_forecast,
                                 "MW", "DE", "24h", physical_min=0.0)
    solar_bundle = ForecastBundle("native solar XGBoost adapted to OPSD",
                                  solar_forecast, "MW", "DE", "24h",
                                  metadata={"capacity_mw": bound},
                                  physical_min=0.0, physical_max=bound)
    month = stamp.dt.tz_convert("Europe/Berlin").dt.month.to_numpy()
    results = {}
    for name, rho in (("copula", rank_corr), ("independent", 0.0)):
        forecasts, decisions = [], []
        for m in range(1, 13):
            take = month == m
            fused = fuse_forecasts(
                load_bundle.replace_frame(load_forecast.loc[take].reset_index(drop=True)),
                solar_bundle.replace_frame(solar_forecast.loc[take].reset_index(drop=True)),
                n_scenarios=scenarios, seed=seed + m,
                dependence=rho, output_location="DE")
            action = make_decisions(fused)
            forecasts.append(fused.forecast)
            decisions.append({"month": m, "summary": action.summary,
                              "risk_flags": action.risk_flags,
                              "recommendations": action.recommendations})
        combined = pd.concat(forecasts, ignore_index=True)
        if not combined.TIMESTAMP.equals(load_forecast.TIMESTAMP):
            raise ValueError("monthly native fusion timestamps do not match")
        combined.to_csv(output / f"net_load_{name}_q01_q99_2019.csv", index=False)
        (output / f"decisions_{name}_2019.json").write_text(
            json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
        results[name] = _score(actual_net, combined[QUANTILES].to_numpy())
    report = {
        "experiment": "OPSD_DE_two_native_model_trainers_adapted_features",
        "not_full_LangGraph_rerun": True,
        "not_original_GEFCom_feature_space": True,
        "source_input_manifest_sha256": _sha(inputs / "input_manifest.json"),
        "models": {"load": "native load train_gbr, 99 quantile estimators",
                   "solar": "native solar train_xgboost, 99 quantile estimators"},
        "solar_xgboost_estimators": solar_estimators,
        "load_gbr_estimators": load_native.GBR_N_ESTIMATORS,
        "train_hours": len(train_load),
        "early_stop_validation_hours": int(validation.sum()),
        "dependence_calibration_hours": int(dependence.sum()),
        "test_hours": len(test_load),
        "rank_correlation": rank_corr,
        "scenarios_per_hour": scenarios,
        "seed": seed,
        "load": _score(actual_load, load_test),
        "solar": _score(actual_solar, solar_test_mw),
        "net_load": results,
        "caveat": "These are two ported native model trainers on new measured features, not a full agent graph or comparable GEFCom-task evaluation.",
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--load-estimators", type=int, default=50)
    parser.add_argument("--solar-estimators", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run(args.inputs, args.output, scenarios=args.scenarios,
                         seed=args.seed, load_estimators=args.load_estimators,
                         solar_estimators=args.solar_estimators), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fresh full-budget GEFCom load experiments with strict 24-hour inputs.

Existing results/checkpoints remain untouched. This explicitly adapted
experiment uses the native Linear/GBR architectures and production budgets,
not the legacy contemporaneous-weather/lag-1 feature set.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .verify_forecasts import _metrics

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
DEFAULT = BASE / "load_full_budget_24h"
QCOLS = [f"q{i:02d}" for i in range(1, 100)]
LEVELS = np.arange(1, 100) / 100.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    import platform
    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost"):
        packages[package] = importlib.metadata.version(package)
    return {"python": platform.python_version(), **packages}


def native_modules(output):
    sys.path.insert(0, str(ROOT / "agentic/pipeline"))
    config = importlib.import_module("config")
    config.LOG_DIR = str(output / "logs")
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    utils = importlib.import_module("utils")
    native = importlib.import_module("model_training")
    return utils, native


def causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    stamp = pd.to_datetime(frame.TIMESTAMP)
    hour, dow, day = stamp.dt.hour, stamp.dt.dayofweek, stamp.dt.dayofyear
    result = pd.DataFrame({"hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
                           "dow_sin": np.sin(2*np.pi*dow/7), "dow_cos": np.cos(2*np.pi*dow/7),
                           "day_sin": np.sin(2*np.pi*day/365.25), "day_cos": np.cos(2*np.pi*day/365.25),
                           "weekend": (dow >= 5).astype(float)})
    for lag in (25, 49, 169, 337):
        result[f"load_lag{lag}"] = frame.LOAD.shift(lag)
    history = frame.LOAD.shift(25)
    for window in (24, 168):
        result[f"load_mean{window}_ending_lag25"] = history.rolling(window).mean()
        result[f"load_std{window}_ending_lag25"] = history.rolling(window).std()
    weather = [f"w{i}" for i in range(1, 26)]
    # Require all stations; no target/feature interpolation with future values.
    mean_temperature = frame[weather].mean(axis=1, skipna=False)
    result["temperature_mean_lag25"] = mean_temperature.shift(25)
    result["temperature_mean_lag169"] = mean_temperature.shift(169)
    return result


def prepare(output: Path = DEFAULT) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "input_manifest.json").exists():
        return validate_inputs(output)[0]
    utils, _ = native_modules(output)
    source = ROOT / "agentic/dataset/GEFCom2014-L_V2/Load/GEFCom2014_Load_merged.csv"
    raw = pd.read_csv(source)
    if not raw.ZONEID.eq(1).all():
        raise ValueError("expected one load zone")
    parsed, previous = [], None
    for value in raw.TIMESTAMP:
        previous = utils.parse_timestamp(value, previous)
        parsed.append(previous)
    raw.TIMESTAMP = parsed
    if raw.TIMESTAMP.duplicated().any() or not raw.TIMESTAMP.is_monotonic_increasing:
        raise ValueError("source timestamps are not unique/chronological")
    if not raw.TIMESTAMP.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("source does not have continuous hourly observations")
    features = causal_features(raw)
    usable = features.notna().all(axis=1) & raw.LOAD.notna() & raw.LOAD.ge(0)
    bounds = {"train": ("2005-01-01", "2009-01-01"),
              "validation": ("2009-01-01", "2010-01-01"),
              "test": ("2010-01-01", "2010-10-01")}
    records = {}
    for split, (start, end) in bounds.items():
        take = usable & raw.TIMESTAMP.ge(pd.Timestamp(start)) & raw.TIMESTAMP.lt(pd.Timestamp(end))
        table = pd.concat([raw.loc[take, ["TIMESTAMP", "LOAD"]], features.loc[take]], axis=1)
        path = output / f"{split}.csv"
        table.to_csv(path, index=False)
        records[split] = {"rows": len(table), "sha256": sha(path), "start_inclusive": start,
                          "end_exclusive": end, "first_timestamp": str(table.TIMESTAMP.iloc[0]),
                          "last_timestamp": str(table.TIMESTAMP.iloc[-1])}
    if records["test"]["rows"] != 6552 or records["validation"]["rows"] != 8760:
        raise ValueError("unexpected evaluation horizon; do not silently alter common hours")
    manifest = {"experiment": "adapted_full_budget_load_strict24h", "source_path": str(source.relative_to(ROOT)),
                "source_sha256": sha(source), "zone": 1, "unit": "native LOAD; physical MW conversion unverified",
                "horizon_hours": 24, "latest_observed_feature_lag_hours": 25,
                "timestamp_timezone": "competition native timestamps; timezone not invented",
                "feature_columns": list(features), "splits": records,
                "preprocessing": "raw released targets; no outlier replacement or future interpolation; training-only scalers",
                "scope": "new adapted feature-availability experiment, not unchanged GEFCom preprocessing or nine-model ranking"}
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate_inputs(output)
    return manifest


def validate_inputs(output: Path):
    manifest = json.loads((output / "input_manifest.json").read_text())
    if manifest["horizon_hours"] != 24 or manifest["latest_observed_feature_lag_hours"] != 25:
        raise ValueError("feature-availability contract changed")
    tables, last = {}, None
    for split in ("train", "validation", "test"):
        path = output / f"{split}.csv"
        if sha(path) != manifest["splits"][split]["sha256"]:
            raise ValueError("prepared input hash mismatch")
        table = pd.read_csv(path)
        stamp = pd.to_datetime(table.TIMESTAMP)
        if last is not None and stamp.iloc[0] <= last:
            raise ValueError("chronological splits overlap")
        last = stamp.iloc[-1]
        if len(table) != manifest["splits"][split]["rows"] or not np.isfinite(table.drop(columns="TIMESTAMP").to_numpy()).all():
            raise ValueError("invalid prepared input")
        tables[split] = table
    return manifest, tables


def vectorized_loss(prediction, target, quantiles):
    """Same all-quantile pinball objective as the native 99-loop loss."""
    import torch
    q = torch.as_tensor(quantiles, dtype=prediction.dtype, device=prediction.device)
    error = target[:, None] - prediction
    return torch.maximum(q*error, (q-1)*error).mean()


def train(model: str, output: Path = DEFAULT) -> dict:
    if model not in ("Linear", "GBR"):
        raise ValueError("this experiment covers exactly the two journal load models")
    output = output.resolve()
    manifest, tables = validate_inputs(output)
    model_dir = output / model
    model_dir.mkdir(exist_ok=True)
    if (model_dir / "evaluation.json").exists():
        return verify_model(model_dir, output)
    _, native = native_modules(output)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    native.LINEAR_MAX_EPOCHS, native.GBR_N_ESTIMATORS = 300, 50
    native.combined_quantile_loss = vectorized_loss
    features = manifest["feature_columns"]
    scaler = StandardScaler().fit(tables["train"][features])
    matrices = {split: scaler.transform(frame[features]).astype(np.float32) for split, frame in tables.items()}
    targets = {split: frame.LOAD.to_numpy(np.float32) for split, frame in tables.items()}
    # Positive affine target normalization stabilizes Linear's raw-scale
    # optimization; its inverse is applied before scoring or export.
    target_mean = float(targets["train"].mean()) if model == "Linear" else 0.
    target_scale = float(targets["train"].std()) if model == "Linear" else 1.
    configuration = {"model": model, "seed": 42, "training_row_limit": None,
                     "training_rows": len(tables["train"]), "horizon_hours": 24,
                     "input_manifest_sha256": sha(output / "input_manifest.json"),
                     "feature_columns": features, "feature_scaler": "training-only StandardScaler",
                     "target_normalization": {"train_mean": target_mean, "train_std": target_scale},
                     "objective": "mean Q01-Q99 pinball; vectorized native objective",
                     "settings": {"max_epochs": 300, "batch_size": 512, "optimizer": "Adam",
                                  "learning_rate": .001, "weight_decay": .0001, "patience": 20}
                                  if model == "Linear" else {"estimators_per_quantile": 50, "max_depth": 3,
                                                              "learning_rate": .1, "subsample": .8},
                     "environment": environment(),
                     "native_training_source_sha256": sha(ROOT / "agentic/pipeline/model_training.py"),
                     "runner_source_sha256": sha(Path(__file__))}
    (model_dir / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    with (model_dir / "feature_scaler.pkl").open("wb") as stream:
        pickle.dump(scaler, stream)
    print(f"{model}: full {len(tables['train'])} training rows; native production budget", flush=True)
    started = time.monotonic()
    if model == "Linear":
        fitted, val, test = native.train_linear(matrices["train"], (targets["train"]-target_mean)/target_scale,
                                               matrices["validation"], (targets["validation"]-target_mean)/target_scale,
                                               matrices["test"])
        val, test = val*target_scale+target_mean, test*target_scale+target_mean
        checkpoint = model_dir / "model.pt"
        torch.save({"state_dict": fitted.state_dict(), "configuration": configuration}, checkpoint)
    else:
        fitted, val, test = native.train_gbr(matrices["train"], targets["train"],
                                            matrices["validation"], targets["validation"], matrices["test"])
        checkpoint = model_dir / "model.pkl"
        with checkpoint.open("wb") as stream:
            pickle.dump(fitted, stream)
    results = {}
    for split, values in (("validation", val), ("test", test)):
        raw_crossings = int((np.diff(values, axis=1) < -1e-10).sum())
        corrected = np.sort(np.clip(values, 0, None), axis=1)
        frame = pd.DataFrame(corrected, columns=QCOLS)
        frame.insert(0, "actual_LOAD", targets[split])
        frame.insert(0, "TIMESTAMP", tables[split].TIMESTAMP.to_numpy())
        path = model_dir / f"{split}_q01_q99.csv"
        frame.to_csv(path, index=False)
        results[split] = {**_metrics(targets[split], corrected), "raw_crossing_pairs": raw_crossings,
                          "forecast_sha256": sha(path)}
    report = {"experiment": "adapted_full_budget_load_strict24h", "model": model,
              "runtime_seconds": time.monotonic()-started, "training_rows": len(tables["train"]),
              "configuration_sha256": sha(model_dir / "configuration.json"),
              "checkpoint_sha256": sha(checkpoint), "scaler_sha256": sha(model_dir / "feature_scaler.pkl"),
              "unit": manifest["unit"], "metrics": results}
    (model_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    verified = verify_model(model_dir, output)
    print(json.dumps(verified["metrics"]["test"], indent=2), flush=True)
    return verified


def verify_model(model_dir: Path, output: Path) -> dict:
    import torch
    manifest, tables = validate_inputs(output)
    report = json.loads((model_dir / "evaluation.json").read_text())
    configuration = json.loads((model_dir / "configuration.json").read_text())
    checkpoint = model_dir / ("model.pt" if report["model"] == "Linear" else "model.pkl")
    for path, field in ((checkpoint, "checkpoint_sha256"),
                        (model_dir / "configuration.json", "configuration_sha256"),
                        (model_dir / "feature_scaler.pkl", "scaler_sha256")):
        if sha(path) != report[field]:
            raise ValueError(f"changed model artifact: {path}")
    with (model_dir / "feature_scaler.pkl").open("rb") as stream:
        scaler = pickle.load(stream)
    if report["model"] == "Linear":
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fitted = torch.nn.Linear(len(manifest["feature_columns"]), 99)
        fitted.load_state_dict(loaded["state_dict"], strict=True)
        fitted.eval()
    else:
        with checkpoint.open("rb") as stream:
            fitted = pickle.load(stream)
    for split in ("validation", "test"):
        path = model_dir / f"{split}_q01_q99.csv"
        frame = pd.read_csv(path)
        if not frame.TIMESTAMP.equals(tables[split].TIMESTAMP):
            raise ValueError("forecast timestamp alignment fails")
        np.testing.assert_allclose(frame.actual_LOAD, tables[split].LOAD, rtol=1e-6)
        if sha(path) != report["metrics"][split]["forecast_sha256"]:
            raise ValueError("forecast artifact changed")
        measured = _metrics(frame.actual_LOAD.to_numpy(), frame[QCOLS].to_numpy())
        for key in ("pinball", "mae", "coverage_q10_q90", "sharpness_q10_q90"):
            np.testing.assert_allclose(measured[key], report["metrics"][split][key], atol=2e-5, rtol=1e-6)
        X = scaler.transform(tables[split][manifest["feature_columns"]]).astype(np.float32)
        if report["model"] == "Linear":
            with torch.no_grad():
                replay = fitted(torch.from_numpy(X)).numpy()
            normalization = configuration["target_normalization"]
            replay = replay*normalization["train_std"]+normalization["train_mean"]
        else:
            replay = np.column_stack([m.predict(X) for m in fitted])
        replay = np.sort(np.clip(replay, 0, None), axis=1)
        np.testing.assert_allclose(replay, frame[QCOLS].to_numpy(), atol=5e-5, rtol=1e-6)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "Linear", "GBR"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2) if args.stage == "prepare" else "")
    if args.stage != "prepare":
        train(args.stage, args.output)

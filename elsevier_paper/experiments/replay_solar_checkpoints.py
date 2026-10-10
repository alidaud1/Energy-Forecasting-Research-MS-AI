"""Create keyed solar forecasts by replaying saved checkpoints, not retraining.

Historical training software/optimizer states cannot be inferred from weights.
Run individual models in isolated processes; all outputs are journal-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .export_reproducibility import ROOT, SOLAR, sha, PACKAGES
from .audit_solar_statistics import QCOLS, LEVELS

OUTPUT = Path(__file__).resolve().parent / "solar_checkpoint_replay"
MODELS = ("XGBoost", "LightGBM", "CatBoost", "LSTM", "TCN", "PatchTST", "iTransformer", "TFT")
TABULAR = MODELS[:3]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def metrics(y, q):
    e = y[:, None] - q
    return {"pinball_percent_capacity": float(np.maximum(LEVELS*e, (LEVELS-1)*e).mean()*100),
            "mae_percent_capacity": float(np.abs(y-q[:, 49]).mean()*100),
            "coverage_q10_q90_percent": float(((y>=q[:, 9]) & (y<=q[:, 89])).mean()*100),
            "width_percent_capacity": float((q[:, 89]-q[:, 9]).mean()*100)}


def run(name, output=OUTPUT):
    sys.path.insert(0, str(ROOT / "elsevier_paper/.research-deps"))
    torch.set_num_threads(2)
    # Archived TCN weights contain many float32 subnormals. Flushing these
    # negligible values avoids a severe CPU convolution slowdown. A separate
    # real-input check records the maximum effect against ordinary arithmetic.
    if name == "TCN":
        torch.set_flush_denormal(True)
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = SOLAR / f"pipeline/saved_models/{name}_best.{'pkl' if name in TABULAR else 'pt'}"
    splits = SOLAR / f"pipeline/outputs/splits/{name}"
    started = time.monotonic()
    config = {"model": name, "checkpoint_sha256": sha(checkpoint), "mode": "CPU checkpoint replay; not fresh training",
              "historical_environment": "unrecorded; current replay environment is recorded separately",
              "splits": {}, "current_environment": {}}
    if name == "TCN":
        config["cpu_flush_float32_subnormals"] = True
    import importlib.metadata
    for package in PACKAGES:
        try:
            config["current_environment"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            config["current_environment"][package] = None
    if name in TABULAR:
        with checkpoint.open("rb") as stream:
            fitted = pickle.load(stream)
        if len(fitted) != 99:
            raise ValueError("expected 99 quantile estimators")
        config["stored_estimator_parameters"] = [json_safe(m.get_params()) for m in fitted]
        header = pd.read_csv(splits / "train.csv", nrows=2)
        features = [c for c in header if c not in {"ZONEID", "TIMESTAMP", "POWER"}
                    and pd.api.types.is_numeric_dtype(header[c])]
    else:
        source = SOLAR / "colab/solar_gpu_training.py"
        spec = importlib.util.spec_from_file_location("solar_replay_architectures", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = module.MODEL_CONFIG[name]
        features = module.get_feature_cols(pd.read_csv(splits / "zone1/train.csv", nrows=2))
        fitted = module.build_model(name, features, cfg)
        fitted.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        config["architecture_source_sha256"] = sha(source)
        config["source_recipe_not_historical_training_manifest"] = cfg
    config["features"] = features
    (destination / "configuration.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
    report = {"model": name, "checkpoint_sha256": sha(checkpoint), "splits": {}}
    for split in ("val", "test"):
        if name in TABULAR:
            current = pd.read_csv(splits / f"{split}.csv")
            X = current[features].to_numpy(dtype=np.float32)
            values = []
            for m in fitted:
                if name == "CatBoost":
                    values.append(m.predict(X, thread_count=2))
                else:
                    m.set_params(n_jobs=2)
                    values.append(m.predict(X))
            raw = np.column_stack(values)
            config["splits"][split] = {"sha256": sha(splits / f"{split}.csv"), "rows": len(current)}
        else:
            datasets, frames = [], []
            for zone in module.ZONES:
                context = module.load_zone_frame(splits, zone, split, cfg["lookback"])
                datasets.append(module.make_dataset(name, context, features, cfg))
                frames.append(pd.read_csv(splits / f"zone{zone}/{split}.csv"))
            current = pd.concat(frames, ignore_index=True)
            loader = DataLoader(ConcatDataset(datasets), batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
            raw = module.predict(name, fitted, loader, torch.device("cpu"))
            config["splits"][split] = {"zone_file_sha256": [sha(splits / f"zone{z}/{split}.csv") for z in module.ZONES],
                                        "rows": len(current)}
        if raw.shape != (len(current), 99) or not np.isfinite(raw).all():
            raise ValueError("invalid forecast dimensions/values")
        keys = current[["ZONEID", "TIMESTAMP", "POWER"]].copy()
        if keys.duplicated(["ZONEID", "TIMESTAMP"]).any():
            raise ValueError("duplicate zone-hour")
        q = np.sort(np.clip(raw.astype(np.float32), 0, 1), axis=1)
        keys = pd.concat([keys.reset_index(drop=True), pd.DataFrame(q, columns=QCOLS)], axis=1)
        path = destination / f"{split}_keyed_q01_q99.csv"
        keys.to_csv(path, index=False)
        record = {"rows": len(keys), "forecast_sha256": sha(path), "metrics": metrics(keys.POWER.to_numpy(), q),
                  "raw_crossing_pairs": int((np.diff(raw, axis=1)<0).sum())}
        if split == "test":
            archive = SOLAR / f"pipeline/outputs/predictions/{name}_predictions.csv"
            old = pd.read_csv(archive).to_numpy(dtype=np.float32)
            old = np.sort(np.clip(old, 0, 1), axis=1)
            if old.shape != q.shape:
                raise ValueError("archived forecast row count differs")
            record["archive_comparison"] = {"archive_sha256": sha(archive),
                "max_absolute_difference": float(np.abs(q-old).max()),
                "mean_absolute_difference": float(np.abs(q-old).mean()),
                "matches_at_absolute_tolerance_1e_5": bool(np.allclose(q, old, atol=1e-5, rtol=0))}
        report["splits"][split] = record
        print(name, split, json.dumps(record), flush=True)
    (destination / "configuration.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n")
    report["configuration_sha256"] = sha(destination / "configuration.json")
    report["runtime_seconds"] = time.monotonic()-started
    (destination / "replay_report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    run(args.model)

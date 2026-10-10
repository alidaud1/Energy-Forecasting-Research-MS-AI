"""Isolated, fully logged fresh solar training using unchanged source settings.

This module is embedded with the original runner into separate journal
notebooks. It never overwrites the historical pipeline outputs. A new UUID
directory is created for each attempt, including interrupted/failed attempts.
"""
from __future__ import annotations

import contextlib
import ast
import copy
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "lightgbm", "catboost")
QCOLS = [f"q{i:02d}" for i in range(1, 100)]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "torch_build": torch.__version__,
            "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(), "gpu": None}
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": props.name, "memory_bytes": props.total_memory,
                       "compute_capability": [props.major, props.minor]}
    return info


class Tee:
    def __init__(self, original, logfile):
        self.original, self.logfile = original, logfile

    def write(self, message):
        self.original.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.original.flush()
        self.logfile.flush()


def keyed_forecasts(index, predictions):
    predictions = np.asarray(predictions)
    if predictions.shape != (len(index), 99) or not np.isfinite(predictions).all():
        raise ValueError("invalid keyed forecast shape/values")
    keys = index[["ZONEID", "TIMESTAMP", "POWER"]].reset_index(drop=True)
    if keys.duplicated(["ZONEID", "TIMESTAMP"]).any():
        raise ValueError("duplicate solar zone-hour")
    if not np.isfinite(keys.POWER).all():
        raise ValueError("nonfinite target")
    return pd.concat([keys, pd.DataFrame(predictions, columns=QCOLS)], axis=1)


def verify_inputs(runner, split_dir):
    indexes, evidence = {}, {}
    for split in ("train", "val", "test"):
        frames = []
        for zone in runner.ZONES:
            path = split_dir / f"zone{zone}/{split}.csv"
            frame = pd.read_csv(path)
            if not frame.ZONEID.eq(zone).all():
                raise ValueError("wrong zone in split file")
            stamps = pd.to_datetime(frame.TIMESTAMP, errors="raise")
            if stamps.duplicated().any() or not stamps.is_monotonic_increasing:
                raise ValueError("unordered/duplicate zone timestamps")
            if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
                raise ValueError("nonfinite training inputs")
            frames.append(frame[["ZONEID", "TIMESTAMP", "POWER"]])
            evidence[f"zone{zone}/{split}.csv"] = {"sha256": sha(path), "rows": len(frame),
                "first_timestamp": str(frame.TIMESTAMP.iloc[0]), "last_timestamp": str(frame.TIMESTAMP.iloc[-1])}
        indexes[split] = pd.concat(frames, ignore_index=True)
    for zone in runner.ZONES:
        last = None
        for split in ("train", "val", "test"):
            stamps = pd.to_datetime(indexes[split].loc[indexes[split].ZONEID.eq(zone), "TIMESTAMP"])
            if last is not None and stamps.iloc[0] <= last:
                raise ValueError("train/validation/test overlap")
            last = stamps.iloc[-1]
    if len(indexes["val"]) != 6624 or len(indexes["test"]) != 2160:
        raise ValueError("incorrect research evaluation partitions")
    return indexes, evidence


def instrument_epoch_history(runner, source, output):
    """Insert only a history-save callback, retaining native numerical steps."""
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_model")
    inserted = False
    for loop in ast.walk(function):
        if not isinstance(loop, ast.For):
            continue
        for i, statement in enumerate(loop.body):
            if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and isinstance(statement.value.func.value, ast.Name)
                    and statement.value.func.value.id == "history"
                    and statement.value.func.attr == "append"):
                callback = ast.Expr(ast.Call(ast.Name("_journal_record_epoch", ast.Load()),
                                            [ast.Name("history", ast.Load())], []))
                loop.body.insert(i+1, callback)
                inserted = True
                break
    if not inserted:
        raise ValueError("native training history append not found; do not instrument an unknown source")

    def record(history):
        temporary = output / "training_history.csv.tmp"
        pd.DataFrame(history).to_csv(temporary, index=False)
        temporary.replace(output / "training_history.csv")

    runner._journal_record_epoch = record
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    snapshot = output / "instrumented_train_model.py"
    snapshot.write_text(ast.unparse(module)+"\n")
    exec(compile(module, str(snapshot), "exec"), runner.__dict__)
    return snapshot


def run_logged_solar(runner, model, runner_source, drive_base, archive_dir=None):
    if model not in runner.MODEL_CONFIG:
        raise ValueError("unknown neural model")
    if not torch.cuda.is_available():
        raise RuntimeError("Fresh research training requires a Colab GPU; no CPU substitution is made.")
    base = Path(drive_base)
    split_dir = base / f"pipeline/outputs/splits/{model}"
    if not split_dir.exists():
        split_dir = base / f"splits/{model}"
    indexes, evidence = verify_inputs(runner, split_dir)
    identifier = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:10]
    output = base / "journal_repro_runs" / model / identifier
    output.mkdir(parents=True, exist_ok=False)
    (output / "original_runner.py").write_text(runner_source)
    env = environment()
    # This file is an actual new-run observation, never a historical inference.
    (output / "environment.json").write_text(json.dumps(env, indent=2)+"\n")
    (output / "requirements_observed.txt").write_text("\n".join(f"{k}=={v}" for k,v in env["packages"].items() if v)+"\n")
    cfg = copy.deepcopy(runner.MODEL_CONFIG[model])
    resolved = {"model": model, "seed": runner.SEED, "unchanged_source_configuration": cfg,
        "learning_rate": cfg.get("lr", .001), "weight_decay": .0001 if cfg["optimizer"] == "adamw" else .00001,
        "gradient_clip_norm": 1., "mixed_precision": cfg.get("amp", True),
        "quantiles": runner.QUANTILES, "source_sha256": sha(output / "original_runner.py"),
        "input_files": evidence, "mode": "fresh training; no historical checkpoint reused"}
    instrumented = instrument_epoch_history(runner, runner_source, output)
    resolved["training_instrumentation"] = "one atomic per-epoch history-save callback; numerical training steps unchanged"
    resolved["instrumented_train_function_sha256"] = sha(instrumented)
    scaler = base / f"pipeline/outputs/scalers/{model}_scaler.pkl"
    if scaler.exists():
        shutil.copy2(scaler, output / "feature_scaler.pkl")
        resolved["scaler_sha256"] = sha(output / "feature_scaler.pkl")
    else:
        resolved["scaler_status"] = "not supplied; prepared split hashes are verified, raw-to-scaled reproduction is incomplete"
    configuration = output / "configuration.json"
    configuration.write_text(json.dumps(resolved, indent=2)+"\n")
    manifest_path = output / "manifest.json"
    manifest = {"run_id": identifier, "model": model, "status": "started", "artifacts": {}}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"New isolated output: {output}", flush=True)
    try:
        with (output / "training.log").open("w") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                runner.set_seed()
                loaders, targets, features = runner.build_loaders(model, split_dir)
                resolved["features"] = features
                resolved["training_windows"] = len(loaders["train"].dataset)
                configuration.write_text(json.dumps(resolved, indent=2)+"\n")
                fitted = runner.build_model(model, features, cfg).to(torch.device("cuda"))
                checkpoint = output / "model_best.pt"
                fitted, history = runner.train_model(model, fitted, loaders, cfg, checkpoint, torch.device("cuda"))
                histories = pd.DataFrame(history)
                histories.to_csv(output / "training_history.csv", index=False)
                if histories.empty:
                    raise ValueError("fresh training produced no epoch history")
                scores = {}
                for split in ("val", "test"):
                    np.testing.assert_allclose(targets[split], indexes[split].POWER, atol=1e-7, rtol=0)
                    raw = runner.predict(model, fitted, loaders[split], torch.device("cuda"))
                    predictions = runner.sanitize_predictions(raw)
                    keyed = keyed_forecasts(indexes[split], predictions)
                    keyed.to_csv(output / f"{split}_keyed_q01_q99.csv", index=False)
                    validation_loss = (runner.pinball_numpy(targets[split], predictions) if split == "val"
                                       else scores["val"]["Pinball_Loss"])
                    scores[split] = runner.calculate_metrics(model, targets[split], predictions, validation_loss)
                    scores[split]["raw_crossing_pairs"] = int((np.diff(raw, axis=1)<0).sum())
                (output / "metrics.json").write_text(json.dumps(scores, indent=2)+"\n")
                selection = {"epochs_completed": len(histories),
                    "selected_epoch": int(histories.loc[histories.val_loss.idxmin(), "epoch"]),
                    "criterion": "lowest raw validation mean Q01-Q99 pinball; no test-based selection"}
                (output / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2)+"\n")
                # New native weights must also reproduce inference after reload.
                reloaded = runner.build_model(model, features, cfg).to(torch.device("cuda"))
                reloaded.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True)
                replay = runner.sanitize_predictions(runner.predict(model, reloaded, loaders["test"], torch.device("cuda")))
                saved = pd.read_csv(output / "test_keyed_q01_q99.csv")[QCOLS].to_numpy()
                np.testing.assert_allclose(replay, saved, atol=1e-6, rtol=0)
        manifest["status"] = "completed_verified"
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(error).__name__
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["artifacts"] = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                                 for p in sorted(output.iterdir()) if p.is_file() and p != manifest_path}
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    if archive_dir:
        target = Path(archive_dir)
        target.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(target / f"{model}-{identifier}"), "zip", output)
    print(f"Verified fresh run: {output}")
    return output

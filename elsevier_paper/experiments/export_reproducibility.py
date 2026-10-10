"""Export recorded settings separately from reconstructed source recipes.

Never infer a historical environment or HPO outcome from current defaults.
Run: python -m elsevier_paper.experiments.export_reproducibility
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOLAR = ROOT / "agentic_solar/solar-forecasting-GEFCom2014"
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost",
            "lightgbm", "catboost", "langgraph", "pytest")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_assignment(path: Path, variable: str):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{variable} not found in {path}")


def source_tabular_recipe(path: Path, function: str) -> dict:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for item in node.body:
                if isinstance(item, ast.Assign) and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name) and item.value.func.id == "dict":
                    return {k.arg: ast.literal_eval(k.value) for k in item.value.keywords}
    raise ValueError(f"literal tabular recipe not found: {function}")


def run(output: Path = BASE / "reproducibility") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    saved = ROOT / "agentic/pipeline/saved_models"
    manifest_path = saved / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    load = {}
    for model in ("Linear", "GBR"):
        entry = manifest["models"][model]
        revision = next(r for r in entry["revisions"] if r["checkpoint_id"] == entry["active_checkpoint_id"])
        for artifact in revision["artifacts"].values():
            if sha(saved / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"changed recorded checkpoint artifact: {model}")
        config_path = saved / revision["artifacts"]["configuration"]["path"]
        config = json.loads(config_path.read_text())
        load[model] = {"recorded_configuration": config, "manifest_checkpoint_id": entry["active_checkpoint_id"],
                       "configuration_sha256": sha(config_path), "recorded_artifacts": revision["artifacts"],
                       "scope": "reduced-budget verification run; not full-budget research training",
                       "training_subset": "latest 5000 rows of chronological training partition"}
        (output / f"load_{model}_recorded_configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    colab_path = SOLAR / "colab/solar_gpu_training.py"
    tabular_path = SOLAR / "pipeline/model_training.py"
    recipes = literal_assignment(colab_path, "MODEL_CONFIG")
    config_path = SOLAR / "pipeline/config.py"
    scalers = literal_assignment(config_path, "SCALER_CONFIG")
    solar, histories = {}, []
    for model in ("LSTM", "TCN", "PatchTST", "iTransformer", "TFT", "XGBoost", "LightGBM", "CatBoost"):
        tabular = model in ("XGBoost", "LightGBM", "CatBoost")
        source = tabular_path if tabular else colab_path
        recipe = source_tabular_recipe(source, f"train_{model.lower()}") if tabular else recipes[model]
        checkpoint = SOLAR / f"pipeline/saved_models/{model}_best.{'pkl' if tabular else 'pt'}"
        record = {"recipe_evidence": "reconstructed from supplied source; no contemporaneous per-model training configuration manifest",
                  "source_path": str(source.relative_to(ROOT)), "source_sha256": sha(source),
                  "configured_recipe": recipe, "configured_seed": 42, "configured_scaler": scalers[model],
                  "checkpoint_path": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint)}
        split_dir = SOLAR / f"pipeline/outputs/splits/{model}"
        header = pd.read_csv(split_dir / "train.csv", nrows=2)
        record["feature_columns"] = [c for c in header if c not in {"ZONEID", "TIMESTAMP", "POWER"}
                                      and pd.api.types.is_numeric_dtype(header[c])]
        record["feature_count"] = len(record["feature_columns"])
        record["splits"] = {}
        for split in ("train", "val", "test"):
            split_path = split_dir / f"{split}.csv"
            keys = pd.read_csv(split_path, usecols=["TIMESTAMP", "ZONEID"])
            record["splits"][split] = {"rows": len(keys), "sha256": sha(split_path),
                                       "timestamp_min": keys.TIMESTAMP.min(),
                                       "timestamp_max": keys.TIMESTAMP.max()}
        history_path = SOLAR / f"pipeline/outputs/training_history/{model}_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            chosen = history.loc[history.val_loss.idxmin()]
            record["recorded_history"] = {"epochs_completed": len(history), "minimum_validation_epoch": int(chosen.epoch),
                                           "minimum_validation_pinball": float(chosen.val_loss), "sha256": sha(history_path)}
            histories.append({"model": model, **record["recorded_history"]})
        solar[model] = record
    # The ensemble checkpoint contains only a small dictionary, not executable
    # estimator classes. Its exact weights are checked against archived forecasts.
    import pickle
    ensemble_path = SOLAR / "pipeline/saved_models/Ensemble_best.pkl"
    with ensemble_path.open("rb") as stream:
        ensemble = pickle.load(stream)
    pred_dir = SOLAR / "pipeline/outputs/predictions"
    weighted = sum(weight * pd.read_csv(pred_dir / f"{model}_predictions.csv").to_numpy(dtype="float32")
                   for model, weight in ensemble["weights"].items())
    import numpy as np
    actual_ensemble = pd.read_csv(pred_dir / "Ensemble_predictions.csv").to_numpy(dtype="float32")
    np.testing.assert_allclose(weighted, actual_ensemble, atol=2e-7, rtol=0)
    environment = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = "not installed in journal audit environment"
    report = {"load": load, "solar": solar,
              "solar_ensemble_recorded": {**ensemble, "checkpoint_sha256": sha(ensemble_path)},
              "journal_audit_environment_not_historical_training_environment": environment,
              "limits": ["Solar historical package versions/HPO overrides are not established by a contemporaneous manifest.",
                         "Solar expected row indexes are reconstructed sidecars, not original prediction keys.",
                         "GEFCom load includes lag-1 and contemporaneous temperatures; do not label its scores strict 24-hour-ahead forecasts.",
                         "Audit-score reproduction is verified; clean-machine full retraining is not claimed."]}
    (output / "experiment_configuration_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(histories).to_csv(output / "solar_recorded_training_history.csv", index=False)
    return report


if __name__ == "__main__":
    report = run()
    print("Saved recorded load settings, reconstructed solar recipes and environment.")
    print(json.dumps(report["journal_audit_environment_not_historical_training_environment"], indent=2))

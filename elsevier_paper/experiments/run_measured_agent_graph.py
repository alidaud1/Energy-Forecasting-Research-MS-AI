"""Run the eight-node forecasting graph on synchronized measured OPSD inputs.

This is an *adapted* graph: GEFCom-specific node bodies are replaced with
measured-data nodes, while the original graph edges and native GBR/XGBoost
training functions remain in use. It never writes to either GEFCom output
directory. Run load and solar in separate processes to avoid their legacy
top-level ``agents``/``state`` module-name collision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .run_opsd_baseline import _forecast_frame, _score
from .run_opsd_native_subset import _clean_quantiles, _matrix

ROOT = Path(__file__).resolve().parents[2]
QUANTILES = [f"q{i:02d}" for i in range(1, 100)]
SPLITS = ("train", "calibration", "test")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_handoff(inputs: Path, domain: str) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Validate archived split files before any graph node can train."""
    inputs = inputs.resolve()
    manifest_path = inputs / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("area"), manifest.get("timestamp"), manifest.get("forecast_issue")) != (
        "DE", "UTC interval start", "strictly 24 hours before target interval start"
    ):
        raise ValueError("incompatible area, timestamp or forecast-issue contract")
    if manifest.get("weather_forecasts_available") is not False:
        raise ValueError("unexpected weather-forecast availability declaration")
    source = Path(manifest["source"])
    if not source.is_file():
        # Older hand-off manifests record the creator's absolute path.
        # Relocated reproducibility packages may keep the same verified
        # source file in the sibling opsd_source directory.
        source = inputs.parent / "opsd_source" / source.name
    if not source.is_file() or _sha(source) != manifest["source_sha256"]:
        raise ValueError("measured source file is missing or has changed")
    tables = {}
    target = "LOAD" if domain == "load" else "POWER"
    other = "solar" if domain == "load" else "load"
    last_end = None
    for split in SPLITS:
        path = inputs / f"{domain}_{split}.csv"
        peer_path = inputs / f"{other}_{split}.csv"
        hashes = manifest["file_sha256"]
        if _sha(path) != hashes[path.name] or _sha(peer_path) != hashes[peer_path.name]:
            raise ValueError(f"{split} file differs from the archived hand-off")
        table = pd.read_csv(path)
        peer = pd.read_csv(peer_path, usecols=["TIMESTAMP", "AREA"])
        if len(table) != manifest["split_rows"][split] or len(table) != len(peer):
            raise ValueError(f"{split} row count differs from the manifest")
        if not table["TIMESTAMP"].equals(peer["TIMESTAMP"]):
            raise ValueError(f"{split} load/solar UTC times do not match")
        if not (table.AREA.eq("DE").all() and peer.AREA.eq("DE").all()):
            raise ValueError(f"{split} area differs from DE")
        stamp = pd.to_datetime(table.TIMESTAMP, utc=True)
        if stamp.isna().any() or not stamp.is_monotonic_increasing or stamp.duplicated().any():
            raise ValueError(f"{split} timestamps are invalid or duplicated")
        if last_end is not None and stamp.iloc[0] <= last_end:
            raise ValueError("chronological splits overlap")
        last_end = stamp.iloc[-1]
        features = manifest[f"{domain}_features"]
        if target in features or "AREA" in features or "TIMESTAMP" in features:
            raise ValueError("a target or identifier has entered model features")
        if not set(features + [target]).issubset(table.columns):
            raise ValueError(f"{split} feature/target columns are incomplete")
        if not np.isfinite(table[features + [target]].to_numpy(dtype=float)).all():
            raise ValueError(f"{split} contains nonfinite model values")
        if (table[target] < 0).any() or (domain == "solar" and (table[target] > 1).any()):
            raise ValueError(f"{split} target violates physical bounds")
        tables[split] = table
    return manifest, tables


def _graph_builder(domain: str):
    project = (ROOT / "agentic" if domain == "load" else
               ROOT / "agentic_solar" / "solar-forecasting-GEFCom2014")
    sys.path.insert(0, str(project))
    return importlib.import_module("graph.pipeline_graph").build_graph


def _native_trainer(domain: str):
    project = (ROOT / "agentic" / "pipeline" if domain == "load" else
               ROOT / "agentic_solar" / "solar-forecasting-GEFCom2014")
    sys.path.insert(0, str(project))
    module = "model_training" if domain == "load" else "pipeline.model_training"
    if domain == "solar":
        return importlib.import_module(module)
    # The load graph has already imported agentic/config.py as the top-level
    # ``config`` module; the legacy native trainer expects pipeline/config.py
    # under that same name. Resolve the trainer once with its own config, then
    # restore the graph's module so later agent imports remain unchanged.
    previous = sys.modules.pop("config", None)
    try:
        return importlib.import_module(module)
    finally:
        sys.modules.pop("config", None)
        if previous is not None:
            sys.modules["config"] = previous


def run(inputs: Path, output: Path, domain: str, *, seed: int = 2026,
        estimators: int = 50) -> dict:
    if domain not in {"load", "solar"} or estimators < 2:
        raise ValueError("domain must be load/solar and estimators >= 2")
    inputs, output = inputs.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    context: dict = {"domain": domain, "inputs": inputs, "output": output}

    def stage(state: dict, name: str, next_step: str, **fields) -> dict:
        return {**state, **fields, "current_step": next_step,
                "completed_steps": [*state.get("completed_steps", []), name]}

    def orchestrator(state: dict) -> dict:
        return state

    def validator(state: dict) -> dict:
        manifest, tables = validate_handoff(inputs, domain)
        context.update(manifest=manifest, tables=tables)
        report = {"valid": True, "area": "DE", "source_sha256": manifest["source_sha256"],
                  "split_rows": manifest["split_rows"]}
        return stage(state, "dataset_validator", "data", validation_report=report)

    def data(state: dict) -> dict:
        target = "LOAD" if domain == "load" else "POWER"
        tables = context["tables"]
        report = {split: {"rows": len(table), "target_min": float(table[target].min()),
                          "target_max": float(table[target].max())}
                  for split, table in tables.items()}
        return stage(state, "data", "feature_eng", data_quality_report=report)

    def feature_eng(state: dict) -> dict:
        features = context["manifest"][f"{domain}_features"]
        for table in context["tables"].values():
            _matrix(table, features)
        path = output / "feature_contract.json"
        path.write_text(json.dumps({"domain": domain, "features": features,
                                    "issue_horizon": "24h", "area": "DE",
                                    "weather_forecasts_available": False}, indent=2) + "\n")
        return stage(state, "feature_eng", "stats", features_path=str(path))

    def stats(state: dict) -> dict:
        target = "LOAD" if domain == "load" else "POWER"
        series = context["tables"]["train"][target]
        report = {"train_mean": float(series.mean()), "train_std": float(series.std()),
                  "train_min": float(series.min()), "train_max": float(series.max())}
        (output / "training_statistics.json").write_text(json.dumps(report, indent=2) + "\n")
        return stage(state, "stats", "training", stats_justifications=[
            "Training-only descriptive statistics; no test-target inspection."])

    def training(state: dict) -> dict:
        manifest, tables = context["manifest"], context["tables"]
        cols = manifest[f"{domain}_features"]
        train, cal, test = (tables[split] for split in SPLITS)
        time = pd.to_datetime(cal.TIMESTAMP, utc=True)
        early = (time < pd.Timestamp("2018-10-01T00:00:00Z")).to_numpy()
        dependence = ~early
        if min(early.sum(), dependence.sum()) < 1000:
            raise ValueError("validation/dependence blocks are too short")
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        native = _native_trainer(domain)
        np.random.seed(seed)
        if domain == "load":
            native.GBR_N_ESTIMATORS = estimators
            models, cal_raw, test_raw = native.train_gbr(
                _matrix(train, cols), train.LOAD.to_numpy(np.float32),
                _matrix(cal, cols), cal.LOAD.to_numpy(np.float32), _matrix(test, cols))
            checkpoint = checkpoint_dir / "Load_GBR_OPSD_DE.pkl"
            with checkpoint.open("wb") as handle:
                pickle.dump(models, handle)
            cal_mw = _clean_quantiles(cal_raw)
            test_mw = _clean_quantiles(test_raw)
            actual_cal = cal.LOAD.to_numpy()
            actual_test = test.LOAD.to_numpy()
            model_name = "GBR"
        else:
            native.MODEL_DIR = str(checkpoint_dir)
            test_raw, early_raw = native.train_xgboost(
                _matrix(train, cols), train.POWER.to_numpy(np.float32),
                _matrix(cal.loc[early], cols), cal.loc[early, "POWER"].to_numpy(np.float32),
                _matrix(test, cols), hpo_params={"n_estimators": estimators, "max_depth": 4})
            checkpoint = checkpoint_dir / "XGBoost_best.pkl"
            with checkpoint.open("rb") as handle:
                models = pickle.load(handle)
            dep_raw = np.column_stack([model.predict(_matrix(cal.loc[dependence], cols))
                                       for model in models])
            cal_mw = np.empty((len(cal), 99), dtype=float)
            cal_mw[early] = _clean_quantiles(early_raw, upper=1.0)
            cal_mw[dependence] = _clean_quantiles(dep_raw, upper=1.0)
            cal_mw *= cal.capacity_lag25.to_numpy()[:, None]
            test_mw = _clean_quantiles(test_raw, upper=1.0) * test.capacity_lag25.to_numpy()[:, None]
            actual_cal = cal.POWER.to_numpy() * cal.capacity_lag25.to_numpy()
            actual_test = test.POWER.to_numpy() * test.capacity_lag25.to_numpy()
            model_name = "XGBoost"
        cal_frame = _forecast_frame(pd.to_datetime(cal.TIMESTAMP, utc=True), cal_mw)
        test_frame = _forecast_frame(pd.to_datetime(test.TIMESTAMP, utc=True), test_mw)
        cal_frame.to_csv(output / "calibration_q01_q99.csv", index=False)
        test_frame.to_csv(output / "test_q01_q99.csv", index=False)
        pd.DataFrame({"TIMESTAMP": cal_frame.TIMESTAMP,
                      "actual_mw": actual_cal,
                      "q50_mw": cal_mw[:, 49]}).to_csv(output / "calibration_residuals.csv", index=False)
        pd.DataFrame({"TIMESTAMP": test_frame.TIMESTAMP,
                      "actual_mw": actual_test}).to_csv(output / "test_actuals.csv", index=False)
        val_score = _score(actual_cal[early], cal_mw[early])
        report = {"model": model_name, "estimators": estimators,
                  "early_validation_hours": int(early.sum()),
                  "dependence_calibration_hours": int(dependence.sum()),
                  "validation": val_score, "checkpoint_sha256": _sha(checkpoint)}
        (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
        context["model_name"] = model_name
        return stage(state, "training", "human_review",
                     training_reports={model_name: report},
                     validation_results={model_name: val_score},
                     checkpoints={model_name: str(checkpoint)})

    def human_review(state: dict) -> dict:
        # The experimental graph is unattended. Do not describe this event as
        # actual human approval or use it to claim a human-in-the-loop study.
        event = {"decision": "approve", "actor": "scripted_experiment",
                 "reason": "continue after measured validation diagnostics",
                 "utc": datetime.now(timezone.utc).isoformat()}
        (output / "review_event.json").write_text(json.dumps(event, indent=2) + "\n")
        return stage(state, "human_review", "evaluation", human_decision="approve",
                     hitl_notes="Scripted experimental approval; not operator review.")

    def evaluation(state: dict) -> dict:
        predictions = pd.read_csv(output / "test_q01_q99.csv")
        actuals = pd.read_csv(output / "test_actuals.csv")
        if not predictions.TIMESTAMP.equals(actuals.TIMESTAMP):
            raise ValueError("evaluation timestamps differ")
        result = _score(actuals.actual_mw.to_numpy(), predictions[QUANTILES].to_numpy())
        report = {"experiment": "adapted_eight_node_measured_forecast_graph",
                  "domain": domain, "model": context["model_name"], "area": "DE",
                  "horizon": "24h", "unit": "MW", "seed": seed,
                  "test_hours": len(predictions), "metrics": result,
                  "scripted_review_not_human": True,
                  "source_input_manifest_sha256": _sha(inputs / "input_manifest.json"),
                  "forecast_sha256": _sha(output / "test_q01_q99.csv"),
                  "checkpoint_sha256": _sha(Path(state["checkpoints"][context["model_name"]]))}
        path = output / "graph_evaluation.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        return stage(state, "evaluation", "done", evaluation_results=report,
                     final_report_path=str(path))

    overrides = {"orchestrator": orchestrator, "dataset_validator": validator,
                 "data": data, "feature_eng": feature_eng, "stats": stats,
                 "training": training, "human_review": human_review,
                 "evaluation": evaluation}
    graph = _graph_builder(domain)(use_memory=False, node_overrides=overrides)
    final = graph.invoke({"current_step": "dataset_validator", "completed_steps": [],
                          "human_decision": "", "errors": []})
    expected = ["dataset_validator", "data", "feature_eng", "stats",
                "training", "human_review", "evaluation"]
    if final.get("current_step") != "done" or final.get("completed_steps") != expected:
        raise RuntimeError("measured graph did not complete all seven non-routing nodes")
    return json.loads((output / "graph_evaluation.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=["load", "solar"])
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--estimators", type=int, default=50)
    args = parser.parse_args()
    result = run(args.inputs, args.output, args.domain,
                 seed=args.seed, estimators=args.estimators)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded validation-driven training, from measured raw CSV to forecasts.

Run each domain in a separate process because the legacy graph modules use
colliding top-level names. Both executors use identical node bodies and policy.
This adapter is deterministic unattended control, not an LLM reasoning study.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .make_opsd_agent_inputs import build as build_inputs
from .prepare_opsd import REQUIRED, prepare
from .run_measured_agent_graph import _graph_builder, _native_trainer, validate_handoff
from .run_opsd_baseline import _forecast_frame, _score
from .run_opsd_native_subset import _clean_quantiles, _matrix

BASE = Path(__file__).resolve().parent
BUDGETS = (10, 50)
GATE_RATIO = .95


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual_mw(table, domain):
    """Never mutate pandas-backed target arrays (read-only with pandas 3)."""
    target = table["LOAD" if domain == "load" else "POWER"].to_numpy(dtype=float, copy=True)
    return target if domain == "load" else target * table.capacity_lag25.to_numpy(dtype=float)


def policy(scores: list[float], reference: float, budgets=BUDGETS, ratio=GATE_RATIO):
    """Validation-only bounded budget controller; lowest score wins, ties first.

    The gate allocates training effort, not permission to operate the grid.
    Exhaustion selects the best available diagnostic forecast and flags a
    missed target. It is never disguised as human approval or dispatch safety.
    """
    if not scores or len(scores) > len(budgets) or not np.isfinite(scores).all():
        raise ValueError("finite validation scores within the declared budget are required")
    if not np.isfinite(reference) or reference < 0 or not 0 < ratio <= 1:
        raise ValueError("invalid reference or gate ratio")
    selected = int(np.argmin(scores))
    gate_met = scores[selected] <= ratio * reference
    return {"action": "retrain" if not gate_met and len(scores) < len(budgets) else "select",
            "selected_index": selected, "gate_met": bool(gate_met),
            "next_budget": budgets[len(scores)] if not gate_met and len(scores) < len(budgets) else None,
            "target_missed_after_budget": bool(not gate_met and len(scores) == len(budgets))}


class TrainingRun:
    def __init__(self, domain, output, source, source_hash, *, trainer=None):
        self.domain, self.output, self.source = domain, Path(output), Path(source)
        self.source_hash = source_hash
        self.output.mkdir(parents=True, exist_ok=False)
        self.trainer = trainer
        self.tables = {}
        self.trials = []
        self.events = []
        self.feature_names = []

    def event(self, stage, action, **detail):
        event = {"sequence": len(self.events), "stage": stage, "action": action, **detail}
        self.events.append(event)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def stage(self, state, name, next_step, **fields):
        self.event(name, "completed", next_step=next_step)
        return {**state, **fields, "current_step": next_step,
                "completed_steps": [*state.get("completed_steps", []), name]}

    def orchestrator(self, state):
        # Domain-specific legacy parser/LLM routing is deliberately not used.
        # A state route cannot skip preconditions or request unlimited training.
        self.event("orchestrator", "route", next_step=state["current_step"])
        return state

    def validator(self, state):
        if sha(self.source) != self.source_hash:
            raise ValueError("raw measured input hash differs from the declared source")
        frame = pd.read_csv(self.source)
        if not set(REQUIRED).issubset(frame.columns) or len(frame) != 26280:
            raise ValueError("raw measured input schema/row count differs")
        return self.stage(state, "dataset_validator", "data", validation_report={"raw_hash_verified": True})

    def data(self, state):
        report = prepare(self.source, self.output / "clean.csv")
        return self.stage(state, "data", "feature_eng", data_quality_report=report)

    def feature_eng(self, state):
        inputs = self.output / "inputs"
        build_inputs(self.output / "clean.csv", inputs)
        self.manifest, self.tables = validate_handoff(inputs, self.domain)
        self.feature_names = self.manifest[f"{self.domain}_features"]
        return self.stage(state, "feature_eng", "stats", features_path=str(inputs))

    def stats(self, state):
        train, validation = self.tables["train"], self.tables["calibration"]
        self.early = (pd.to_datetime(validation.TIMESTAMP, utc=True) < pd.Timestamp("2018-10-01T00:00:00Z")).to_numpy()
        self.dependence = ~self.early
        if min(self.early.sum(), self.dependence.sum()) < 1000:
            raise ValueError("insufficient validation/dependence observations")
        target = "LOAD" if self.domain == "load" else "POWER"
        actual = validation.loc[self.early, target].to_numpy()
        reference = validation.loc[self.early, "load_lag25" if self.domain == "load" else "solar_ratio_lag25"].to_numpy()
        if self.domain == "solar":
            cap = validation.loc[self.early, "capacity_lag25"].to_numpy()
            actual, reference = actual * cap, reference * cap
        self.reference_score = _score(actual, np.repeat(reference[:, None], 99, axis=1))["mean_pinball_loss_mw"]
        report = {"training_mean": float(train[target].mean()), "training_std": float(train[target].std()),
                  "validation_reference": "lag25 point persistence repeated over quantiles",
                  "validation_reference_pinball_mw": self.reference_score,
                  "gate_ratio": GATE_RATIO, "budgets": BUDGETS}
        (self.output / "training_statistics.json").write_text(json.dumps(report, indent=2) + "\n")
        return self.stage(state, "stats", "training", stats_justifications=["bounded validation-only effort allocation"])

    def training(self, state):
        budget = BUDGETS[len(self.trials)]
        folder = self.output / f"trial_{budget}"
        folder.mkdir(exist_ok=False)
        native = self.trainer or _native_trainer(self.domain)
        # Isolate the native functions' log output and checkpoint destinations.
        native.logger = logging.getLogger(f"autonomy_{self.domain}")
        native.SEED = 42
        np.random.seed(42)
        train = self.tables["train"]
        validation = self.tables["calibration"].loc[self.early]
        target = "LOAD" if self.domain == "load" else "POWER"
        x_train, x_val = (_matrix(frame, self.feature_names) for frame in (train, validation))
        y_train, y_val = (frame[target].to_numpy(np.float32) for frame in (train, validation))
        start = time.perf_counter()
        if self.domain == "load":
            native.GBR_N_ESTIMATORS = budget
            models, raw, _ = native.train_gbr(x_train, y_train, x_val, y_val, x_val)
            checkpoint = folder / "models.pkl"
            with checkpoint.open("wb") as handle:
                pickle.dump(models, handle)
        else:
            native.MODEL_DIR = str(folder)
            _, raw = native.train_xgboost(x_train, y_train, x_val, y_val, x_val,
                                        hpo_params={"n_estimators": budget, "max_depth": 4})
            checkpoint = folder / "XGBoost_best.pkl"
        elapsed = time.perf_counter() - start
        with checkpoint.open("rb") as handle:
            models = pickle.load(handle)
        restored = np.column_stack([model.predict(x_val) for model in models]).astype(np.float32)
        np.testing.assert_array_equal(restored, raw)
        values = _clean_quantiles(raw, upper=1 if self.domain == "solar" else np.inf)
        actual = validation[target].to_numpy()
        if self.domain == "solar":
            cap = validation.capacity_lag25.to_numpy()
            values, actual = values * cap[:, None], actual * cap
        score = _score(actual, values)
        _forecast_frame(pd.to_datetime(validation.TIMESTAMP, utc=True), values).to_csv(folder / "validation_q01_q99.csv", index=False)
        config = {"domain": self.domain, "model": "GBR" if self.domain == "load" else "XGBoost",
                  "seed": 42, "maximum_estimators": budget,
                  "actual_boosting_rounds": [int(model.get_booster().num_boosted_rounds()) if hasattr(model, "get_booster")
                                           else int(len(model.estimators_)) for model in models],
                  "features": self.feature_names, "validation": score, "training_seconds": elapsed,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                  "checkpoint_reload_verified": True, "test_input_passed_to_trainer": False,
                  "estimator_parameters": models[0].get_params(), "scaler": "none: native tabular inputs"}
        (folder / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
        self.trials.append(config)
        self.event("training", "fitted", budget=budget, validation_pinball_mw=score["mean_pinball_loss_mw"])
        return self.stage(state, "training", "human_review", training_reports={"candidate": config})

    def review(self, state):
        decision = policy([trial["validation"]["mean_pinball_loss_mw"] for trial in self.trials], self.reference_score)
        self.selection = decision
        self.event("validation_controller", decision["action"], **{k:v for k,v in decision.items() if k != "action"},
                   actor="deterministic_bounded_controller_not_human_or_LLM")
        if decision["action"] == "retrain":
            return self.stage(state, "human_review", "training", human_decision="retrain",
                              hitl_notes="Adapted validation controller, not human approval")
        self.selected = self.trials[decision["selected_index"]]
        return self.stage(state, "human_review", "evaluation", human_decision="approve",
                          hitl_notes="Automatic diagnostic forecast selection; not dispatch authorization",
                          checkpoints={"selected": self.selected["checkpoint"]})

    def predict(self, trial, table):
        with Path(trial["checkpoint"]).open("rb") as handle:
            models = pickle.load(handle)
        raw = np.column_stack([model.predict(_matrix(table, self.feature_names)) for model in models]).astype(np.float32)
        values = _clean_quantiles(raw, upper=1 if self.domain == "solar" else np.inf)
        if self.domain == "solar":
            values *= table.capacity_lag25.to_numpy()[:, None]
        return values

    def evaluation(self, state):
        # Test targets are used for scores only AFTER selection is frozen.
        # Earlier schema/physical-bound validation is not model selection.
        test, cal = self.tables["test"], self.tables["calibration"]
        for name, trial in (("selected", self.selected), ("fixed_initial", self.trials[0])):
            values = self.predict(trial, test)
            _forecast_frame(pd.to_datetime(test.TIMESTAMP, utc=True), values).to_csv(self.output / f"{name}_test_q01_q99.csv", index=False)
            dep = cal.loc[self.dependence]
            dep_values = self.predict(trial, dep)
            _forecast_frame(pd.to_datetime(dep.TIMESTAMP, utc=True), dep_values).to_csv(self.output / f"{name}_dependence_q01_q99.csv", index=False)
        dep = cal.loc[self.dependence]
        values = self.predict(self.selected, dep)
        _forecast_frame(pd.to_datetime(dep.TIMESTAMP, utc=True), values).to_csv(self.output / "dependence_q01_q99.csv", index=False)
        dep_actual = actual_mw(dep, self.domain)
        test_actual = actual_mw(test, self.domain)
        pd.DataFrame({"TIMESTAMP": dep.TIMESTAMP, "actual_mw": dep_actual, "q50_mw": values[:,49]}).to_csv(self.output / "dependence_residuals.csv", index=False)
        report = {"domain": self.domain, "selected_budget": self.selected["maximum_estimators"],
            "trials": [{"budget": trial["maximum_estimators"], "validation": trial["validation"],
                        "training_seconds": trial["training_seconds"], "checkpoint_sha256": trial["checkpoint_sha256"]} for trial in self.trials],
            "selection": self.selection, "reference_validation_pinball_mw": self.reference_score,
            "split_rows": self.manifest["split_rows"], "test_hours": len(test),
            "scripted_human_approval": False, "controller": "deterministic_bounded_validation_policy",
            "unrestricted_or_LLM_autonomy": False, "test_used_for_selection": False,
            "selected_test": _score(test_actual, self.predict(self.selected, test)),
            "fixed_initial_test": _score(test_actual, self.predict(self.trials[0], test))}
        path = self.output / "evaluation.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        return self.stage(state, "evaluation", "done", evaluation_results=report, final_report_path=str(path))

    def nodes(self):
        return {"orchestrator": self.orchestrator, "dataset_validator": self.validator,
                "data": self.data, "feature_eng": self.feature_eng, "stats": self.stats,
                "training": self.training, "human_review": self.review, "evaluation": self.evaluation}


def plain_execute(nodes, initial):
    state = initial
    step = "orchestrator"
    for _ in range(40):
        state = nodes[step](state)
        if step == "orchestrator":
            step = state["current_step"]
        elif step in {"dataset_validator", "data", "feature_eng", "stats", "evaluation"}:
            step = "orchestrator"
        elif step == "training":
            step = "human_review"
        else:
            step = "training" if state["human_decision"] == "retrain" else "evaluation"
            if state["human_decision"] == "abort":
                return state
        if step in {"done", "abort", "error"}:
            return state
    raise RuntimeError("bounded control exceeded its declared step limit")


def run_domain(domain, output=None):
    if domain not in {"load", "solar"}:
        raise ValueError("domain must be load or solar")
    output = Path(output or BASE / "training_autonomy" / domain)
    output.mkdir(parents=True, exist_ok=True)
    source_snapshot = Path(__file__).read_bytes()
    (output / "runner_source.py").write_bytes(source_snapshot)
    source = BASE / "opsd_source/de_2017_2019.csv"
    source_hash = json.loads((BASE / "opsd_source/de_2017_2019_clean_manifest.json").read_text())["source_sha256"]
    protocol = {"domain": domain, "source_sha256": source_hash, "budgets": BUDGETS,
        "gate_ratio": GATE_RATIO, "seed": 42, "validation_end_exclusive": "2018-10-01T00:00:00Z",
        "policy": "expand budget only when validation loss misses 0.95 times lag25 persistence; select minimum validation loss within bounded trials",
        "test": "2019 reused chronological test, not a new independent holdout",
        "executors": ["python", "langgraph"], "same_node_bodies_and_policy": True,
        "scope": "adapted native two-model CPU study, not full nine-model or LLM-agent superiority",
        "fixed_initial": "counterfactual using the initial trial, not another timed executor",
        "no_API_calls": True, "externally_preregistered": False}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    builder = _graph_builder(domain)
    reports = {}
    for engine in np.random.default_rng(2026).permutation(["python", "langgraph"]):
        runner = TrainingRun(domain, output / engine, source, source_hash)
        initial = {"current_step": "dataset_validator", "completed_steps": [], "errors": [], "human_decision": ""}
        compilation = time.perf_counter()
        graph = builder(use_memory=False, node_overrides=runner.nodes()) if engine == "langgraph" else None
        compilation = time.perf_counter() - compilation
        started = time.perf_counter()
        final = graph.invoke(initial, config={"recursion_limit": 40}) if graph else plain_execute(runner.nodes(), initial)
        elapsed = time.perf_counter() - started
        if final["current_step"] != "done" or len(runner.trials) > len(BUDGETS):
            raise RuntimeError("training controller failed its completion/budget contract")
        report = json.loads((runner.output / "evaluation.json").read_text())
        report.update(executor=engine, wall_seconds=elapsed, compilation_seconds=compilation,
                      completed_steps=final["completed_steps"], audit_events=len(runner.events))
        (runner.output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
        reports[engine] = report
    for name in ("selected_test_q01_q99.csv", "fixed_initial_test_q01_q99.csv", "dependence_q01_q99.csv",
                 "selected_dependence_q01_q99.csv", "fixed_initial_dependence_q01_q99.csv"):
        left, right = (pd.read_csv(output / engine / name) for engine in ("python", "langgraph"))
        assert left.TIMESTAMP.equals(right.TIMESTAMP)
        np.testing.assert_array_equal(left.iloc[:,1:], right.iloc[:,1:])
    for key in ("selected_budget", "selection", "selected_test", "fixed_initial_test"):
        assert reports["python"][key] == reports["langgraph"][key]
    assert [json.loads(line) for line in (output / "python/events.jsonl").read_text().splitlines()] == [
        json.loads(line) for line in (output / "langgraph/events.jsonl").read_text().splitlines()]
    result = {"protocol": protocol, "reports": reports, "paired_forecasts_and_selection_exact": True,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "langgraph")}},
        "script_sha256": hashlib.sha256(source_snapshot).hexdigest()}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("load", "solar"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_domain(args.domain, args.output)
    print(json.dumps({key: value for key,value in result.items() if key not in {"environment"}}, indent=2))

"""Matched post-forecast LangGraph versus ordinary Python execution.

Both call the SAME stage functions, validations, recovery and audit writer.
This is not a comparison of LLM reasoning or fresh model training.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph

from fusion.decision_agent import make_decisions
from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS
from fusion.fusion_agent import fuse_forecasts

BASE = Path(__file__).resolve().parent
Q = list(QUANTILE_COLUMNS)
STAGES = ("read", "validate", "fusion", "decision")
CASE_NAMES = ("clean", "missing_quantile", "nonfinite_quantile", "crossing_quantiles",
              "duplicate_timestamp", "reversed_timestamps", "invalid_timestamp",
              "no_timestamp_overlap", "horizon_mismatch", "unsupported_unit",
              "declared_kW_conversion", "unverified_geographic_label",
              "transient_fusion_failure", "persistent_fusion_failure")


class State(TypedDict, total=False):
    load: object
    solar: object
    capacity: float
    case: str
    unit: str
    horizon: str
    location: str
    scenarios: int
    seed: int
    events: list
    error: str
    audit_path: str
    bundles: object
    fused: object
    decision: object


def _event(state, stage, action, detail=""):
    event = {"sequence": len(state["events"]), "stage": stage,
             "action": action, "detail": detail}
    state["events"] = [*state["events"], event]
    if state.get("audit_path"):
        with Path(state["audit_path"]).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


def execute_stage(state: State, stage: str) -> State:
    """Identical stage implementation used by both orchestration engines."""
    state = dict(state)
    if state.get("error"):
        return state
    _event(state, stage, "started")
    try:
        if stage == "read":
            state["load"], state["solar"] = state["load"].copy(), state["solar"].copy()
        elif stage == "validate":
            # Safe canonical ordering is permitted in BOTH implementations.
            # No imputation, invented quantiles or changed horizon is allowed.
            solar = state["solar"]
            stamp = pd.to_datetime(solar.TIMESTAMP, errors="coerce")
            if stamp.notna().all() and not stamp.duplicated().any() and not stamp.is_monotonic_increasing:
                solar = solar.assign(TIMESTAMP=stamp).sort_values("TIMESTAMP").reset_index(drop=True)
                _event(state, stage, "recovered", "canonical timestamp ordering")
            scale = 1000 if state["unit"] == "kW" else 1
            load = ForecastBundle("measured load", state["load"], "MW", "DE", "24h", physical_min=0)
            solar = ForecastBundle("measured solar", solar, state["unit"], state["location"],
                                   state["horizon"], metadata={"capacity_mw": state["capacity"]},
                                   physical_min=0, physical_max=state["capacity"] * scale)
            state["bundles"] = (load, solar)
        elif stage == "fusion":
            for attempt in (1, 2):
                try:
                    if state["case"] == "persistent_fusion_failure" or (
                        state["case"] == "transient_fusion_failure" and attempt == 1):
                        raise TimeoutError("controlled injected failure")
                    state["fused"] = fuse_forecasts(*state["bundles"],
                        n_scenarios=state["scenarios"], seed=state["seed"],
                        dependence=-.25, output_location="DE")
                    if attempt == 2:
                        _event(state, stage, "recovered", "successful bounded retry")
                    break
                except TimeoutError:
                    if attempt == 2:
                        raise
                    _event(state, stage, "retry", "one bounded retry after injected timeout")
        elif stage == "decision":
            state["decision"] = make_decisions(state["fused"])
        else:
            raise ValueError("unknown stage")
        _event(state, stage, "completed")
    except (ValueError, KeyError, TypeError, TimeoutError) as error:
        state["error"] = f"{type(error).__name__}: {error}"
        _event(state, stage, "rejected", state["error"])
    return state


def build_graph():
    graph = StateGraph(State)
    for stage in STAGES:
        graph.add_node(stage, lambda state, name=stage: execute_stage(state, name))
    graph.set_entry_point(STAGES[0])
    for left, right in zip(STAGES, STAGES[1:]):
        graph.add_conditional_edges(left, lambda state, nxt=right: END if state.get("error") else nxt,
                                    {END: END, right: right})
    graph.add_edge(STAGES[-1], END)
    return graph.compile()


def python_pipeline(state):
    for stage in STAGES:
        state = execute_stage(state, stage)
        if state.get("error"):
            break
    return state


def cases(load, solar):
    clean = dict(load=load, solar=solar, unit="MW", horizon="24h", location="DE")
    result = {"clean": clean}
    def add(name, frame=None, **metadata):
        result[name] = {**clean, "solar": solar if frame is None else frame, **metadata}
    add("missing_quantile", solar.drop(columns="q37"))
    frame = solar.copy(); frame.loc[0, "q50"] = np.nan
    add("nonfinite_quantile", frame)
    frame = solar.copy(); frame.loc[0, "q01"] = frame.loc[0, "q99"] + 1
    add("crossing_quantiles", frame)
    frame = solar.copy(); frame.loc[1, "TIMESTAMP"] = frame.loc[0, "TIMESTAMP"]
    add("duplicate_timestamp", frame)
    add("reversed_timestamps", solar.iloc[::-1].reset_index(drop=True))
    frame = solar.copy(); frame["TIMESTAMP"] = frame.TIMESTAMP.astype(object)
    frame.loc[0, "TIMESTAMP"] = "not-a-date"
    add("invalid_timestamp", frame)
    frame = solar.copy(); frame.TIMESTAMP = (pd.to_datetime(frame.TIMESTAMP) + pd.Timedelta(days=400)).astype(str)
    add("no_timestamp_overlap", frame)
    add("horizon_mismatch", horizon="48h")
    add("unsupported_unit", unit="MWh")
    frame = solar.copy(); frame[Q] *= 1000
    add("declared_kW_conversion", frame, unit="kW")
    add("unverified_geographic_label", location="different_grid")
    add("transient_fusion_failure")
    add("persistent_fusion_failure")
    assert tuple(result) == CASE_NAMES
    return result


def verify_audit(state):
    events = state["events"]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    starts = [event["stage"] for event in events if event["action"] == "started"]
    terminals = [event["stage"] for event in events if event["action"] in ("completed", "rejected")]
    assert starts == terminals and starts == list(STAGES[:len(starts)])
    assert bool(state.get("error")) == (events[-1]["action"] == "rejected")
    if state.get("error"):
        assert "fused" not in state and "decision" not in state
    else:
        assert "decision" in state and len(starts) == len(STAGES)
    if state.get("audit_path"):
        assert [json.loads(line) for line in Path(state["audit_path"]).read_text().splitlines()] == events


def run(output: Path, *, repeats=10, hours=72, scenarios=1000):
    if repeats < 2 or hours < 2 or scenarios < 2:
        raise ValueError("repeats, hours and scenarios must be at least two")
    output.mkdir(parents=True, exist_ok=True)
    experiment_id = uuid.uuid4().hex
    audits = output / "audits" / experiment_id
    audits.mkdir(parents=True, exist_ok=False)
    protocol = {"scope": "matched post-forecast orchestration, not model training or LLM reasoning",
        "same_stage_functions": True, "same_validation_recovery_and_audit": True,
        "repeats_per_case_engine": repeats, "hours": hours, "scenarios": scenarios,
        "sampling_seed": 2026, "order_seed": 2026, "fixed_rank_correlation": -.25,
        "timed": "node execution including audit JSONL writes; excludes graph compilation and result export",
        "recovery": ["canonical ordering of unique valid timestamps", "one retry of injected TimeoutError"],
        "no_repairs": "missing/nonfinite/crossed quantiles, duplicate/invalid timestamps, horizon/unit mismatches",
        "geography": "negative control: metadata does not authenticate electrical area",
        "cost": "no LLM/API calls; CPU wall time measured, monetary compute cost unavailable",
        "cases": CASE_NAMES, "not_preregistered": True}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    paths = [BASE / "opsd_measured_full_graph" / domain / "test_q01_q99.csv" for domain in ("load", "solar")]
    load, solar = [pd.read_csv(path).iloc[:hours].copy() for path in paths]
    if len(load) != hours or not load.TIMESTAMP.equals(solar.TIMESTAMP):
        raise ValueError("archived inputs must provide identical requested timestamps")
    capacity = float(pd.read_csv(BASE / "opsd_agent_port/solar_calibration.csv",
                                usecols=["capacity_lag25"]).capacity_lag25.max())
    source_snapshot = Path(__file__).read_bytes()
    (audits / "runner_source.py").write_bytes(source_snapshot)
    start = time.perf_counter(); graph = build_graph(); compilation = time.perf_counter() - start
    engines = {"python": python_pipeline, "langgraph": graph.invoke}
    # Untimed warm-up for both engines, identical data/settings, no disk logger.
    for engine in engines.values():
        engine({**cases(load, solar)["clean"], "case": "clean", "events": [],
                "capacity": capacity, "scenarios": scenarios, "seed": 2026})
    reference = python_pipeline({**cases(load, solar)["clean"], "case": "clean", "events": [],
                                 "capacity": capacity, "scenarios": scenarios, "seed": 2026})
    reference["fused"].forecast.to_csv(output / "clean_net_load_q01_q99.csv", index=False)
    decision = reference["decision"]
    (output / "clean_decision.json").write_text(json.dumps({"summary": decision.summary,
        "recommendations": decision.recommendations, "risk_flags": decision.risk_flags}, indent=2) + "\n")
    rng = np.random.default_rng(2026)
    rows, summaries = [], []
    for name, inputs in cases(load, solar).items():
        pair_outputs = {}
        for repeat in range(repeats):
            for label in rng.permutation(["python", "langgraph"]):
                audit_path = audits / f"{name}_{repeat}_{label}.jsonl"
                state = {**inputs, "case": name, "events": [], "capacity": capacity,
                         "scenarios": scenarios, "seed": 2026, "audit_path": str(audit_path)}
                start = time.perf_counter(); result = engines[label](state); elapsed = time.perf_counter() - start
                verify_audit(result)
                pair_outputs[label] = result
                rows.append({"case": name, "repeat": repeat, "engine": label,
                    "seconds": elapsed, "outcome": "rejected" if result.get("error") else "completed",
                    "recoveries": sum(e["action"] == "recovered" for e in result["events"]),
                    "audit_complete": True, "events": len(result["events"]),
                    "error": result.get("error", ""), "api_calls": 0})
            left, right = pair_outputs["python"], pair_outputs["langgraph"]
            assert left["events"] == right["events"]
            if not left.get("error"):
                np.testing.assert_array_equal(left["fused"].forecast[Q], right["fused"].forecast[Q])
                assert left["decision"] == right["decision"]
                # Repairs and declared conversion must recover the clean output.
                np.testing.assert_allclose(left["fused"].forecast[Q], reference["fused"].forecast[Q], rtol=0, atol=1e-9)
            else:
                assert left["error"] == right["error"]
        summaries.append({"case": name, "outcome": "rejected" if left.get("error") else "completed",
            "same_forecasts_decisions_and_audit": True,
            "recoveries": sum(e["action"] == "recovered" for e in left["events"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "timings.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "cases.csv", index=False)
    timing = {}
    for label in engines:
        values = frame.loc[frame.engine.eq(label) & frame.case.eq("clean"), "seconds"].to_numpy() * 1000
        timing[label] = {"clean_median_ms": float(np.median(values)),
                         "clean_q25_ms": float(np.quantile(values, .25)), "clean_q75_ms": float(np.quantile(values, .75))}
    report = {"protocol": protocol, "experiment_id": experiment_id, "cases": summaries,
        "runs": len(rows), "audit_complete_runs": int(frame.audit_complete.sum()),
        "rejected_cases_per_engine": sum(c["outcome"] == "rejected" for c in summaries),
        "recovered_cases_per_engine": sum(c["recoveries"] > 0 for c in summaries),
        "negative_control_accepted_both": True, "clean_quantile_max_difference_mw": 0.0,
        "timing": timing, "graph_compilation_ms": compilation * 1000,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "langgraph")}},
        "sha256": {str(path.relative_to(BASE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "script_sha256": hashlib.sha256(source_snapshot).hexdigest(),
        "shared_implementation_sha256": {str(path.relative_to(BASE.parents[1])):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (BASE.parents[1] / "fusion" / name for name in ("forecast_contract.py", "fusion_agent.py", "decision_agent.py"))},
        "interpretation": "Shared checks and recovery give identical outcomes; no unique agentic reliability or accuracy benefit established. Local overhead is not a hardware-general estimate."}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (audits / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    report = run(BASE / "matched_orchestration")
    print(json.dumps({key: report[key] for key in ("runs", "audit_complete_runs", "rejected_cases_per_engine",
        "recovered_cases_per_engine", "timing", "graph_compilation_ms", "interpretation")}, indent=2))

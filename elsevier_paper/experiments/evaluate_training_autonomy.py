"""Connect completed bounded training runs to fusion/decision, without retraining."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph
from scipy.stats import spearmanr

from fusion.decision_agent import make_decisions
from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS
from fusion.fusion_agent import fuse_forecasts
from .run_opsd_baseline import _score
from .audit_solar_statistics import hac_mean

BASE = Path(__file__).resolve().parent
Q = list(QUANTILE_COLUMNS)


class JointState(TypedDict, total=False):
    load: object
    solar: object
    dependence: float
    seed: int
    scenarios: int
    fused: object
    decision: object


def fusion_node(state):
    return {"fused": fuse_forecasts(state["load"], state["solar"],
        dependence=state["dependence"], n_scenarios=state["scenarios"], seed=state["seed"], output_location="DE")}


def decision_node(state):
    return {"decision": make_decisions(state["fused"])}


def joint_graph():
    graph = StateGraph(JointState)
    graph.add_node("fusion", fusion_node)
    graph.add_node("decision", decision_node)
    graph.set_entry_point("fusion")
    graph.add_edge("fusion", "decision")
    graph.add_edge("decision", END)
    return graph.compile()


def run(base=None, *, scenarios=1000):
    base = Path(base or BASE / "training_autonomy")
    if scenarios < 2:
        raise ValueError("at least two scenarios are required")
    reports = {domain: json.loads((base / domain / "summary.json").read_text()) for domain in ("load", "solar")}
    assert all(r["paired_forecasts_and_selection_exact"] for r in reports.values())
    assert reports["load"]["protocol"]["source_sha256"] == reports["solar"]["protocol"]["source_sha256"]
    output = base / "joint"
    output.mkdir(exist_ok=True)
    evaluator = joint_graph()
    results, forecasts, decisions = {}, {}, {}
    files = []
    for track, executor, flavor in (("adaptive_python", "python", "selected"),
                                   ("adaptive_langgraph", "langgraph", "selected"),
                                   ("fixed_initial_counterfactual", "python", "fixed_initial")):
        folders = {d: base / d / executor for d in reports}
        prediction_paths = {d: folders[d] / f"{flavor}_test_q01_q99.csv" for d in reports}
        dep_paths = {d: folders[d] / f"{flavor}_dependence_q01_q99.csv" for d in reports}
        prediction = {d: pd.read_csv(path) for d,path in prediction_paths.items()}
        dependence = {d: pd.read_csv(path) for d,path in dep_paths.items()}
        actual_paths = {d: folders[d] / f"inputs/{d}_test.csv" for d in reports}
        dep_actual_paths = {d: folders[d] / "dependence_residuals.csv" for d in reports}
        actual = {d: pd.read_csv(path) for d,path in actual_paths.items()}
        dep_actual = {d: pd.read_csv(path) for d,path in dep_actual_paths.items()}
        files.extend([*prediction_paths.values(), *dep_paths.values(), *actual_paths.values(), *dep_actual_paths.values()])
        for kind in (prediction, dependence, actual, dep_actual):
            assert pd.to_datetime(kind["load"].TIMESTAMP, utc=True).equals(pd.to_datetime(kind["solar"].TIMESTAMP, utc=True))
        for d in reports:
            assert pd.to_datetime(prediction[d].TIMESTAMP, utc=True).equals(pd.to_datetime(actual[d].TIMESTAMP, utc=True))
            assert pd.to_datetime(dependence[d].TIMESTAMP, utc=True).equals(pd.to_datetime(dep_actual[d].TIMESTAMP, utc=True))
        rho = float(spearmanr(dep_actual["load"].actual_mw - dependence["load"].q50,
                              dep_actual["solar"].actual_mw - dependence["solar"].q50).statistic)
        rho = 0.0 if not np.isfinite(rho) else float(np.clip(rho, -.99, .99))
        solar_train = pd.read_csv(folders["solar"] / "inputs/solar_train.csv")
        solar_cal = pd.read_csv(folders["solar"] / "inputs/solar_calibration.csv")
        capacity = float(max(solar_train.capacity_lag25.max(), solar_cal.capacity_lag25.max()))
        stamp = pd.to_datetime(prediction["load"].TIMESTAMP, utc=True)
        month = stamp.dt.tz_convert("Europe/Berlin").dt.month.to_numpy()
        rows, summaries = [], []
        for m in range(1, 13):
            take = month == m
            load = ForecastBundle(f"bounded load {track}", prediction["load"].loc[take].reset_index(drop=True),
                                  "MW", "DE", "24h", physical_min=0)
            solar = ForecastBundle(f"bounded solar {track}", prediction["solar"].loc[take].reset_index(drop=True),
                                   "MW", "DE", "24h", metadata={"capacity_mw": capacity}, physical_min=0, physical_max=capacity)
            initial = {"load": load, "solar": solar, "dependence": rho, "scenarios": scenarios, "seed": 2026 + m}
            if executor == "langgraph":
                state = evaluator.invoke(initial)
            else:
                state = {**initial, **fusion_node(initial)}
                state.update(decision_node(state))
            rows.append(state["fused"].forecast)
            decision = state["decision"]
            summaries.append({"local_month": m, "summary": decision.summary,
                              "recommendations": decision.recommendations, "risk_flags": decision.risk_flags,
                              "scope": "research indicators, not autonomous grid dispatch"})
        forecasts[track] = pd.concat(rows, ignore_index=True).sort_values("TIMESTAMP").reset_index(drop=True)
        # Restore actual order by the same UTC-keyed timestamps, not positional guesses.
        actual_net = actual["load"].LOAD.to_numpy() - actual["solar"].POWER.to_numpy() * actual["solar"].capacity_lag25.to_numpy()
        actual_frame = pd.DataFrame({"TIMESTAMP": stamp.dt.tz_localize(None).to_numpy(), "net_mw": actual_net})
        keyed = forecasts[track].merge(actual_frame, on="TIMESTAMP", how="inner", validate="one_to_one")
        assert len(keyed) == 8760
        results[track] = {"net_load": _score(keyed.net_mw.to_numpy(), keyed[Q].to_numpy()),
                          "rank_correlation": rho, "scenarios_per_hour": scenarios, "selected_budget":
                          {d: (reports[d]["reports"][executor]["selected_budget"] if flavor == "selected" else 10) for d in reports}}
        forecasts[track].to_csv(output / f"{track}_net_q01_q99.csv", index=False)
        (output / f"{track}_decisions.json").write_text(json.dumps(summaries, indent=2) + "\n")
        decisions[track] = summaries
    np.testing.assert_array_equal(forecasts["adaptive_python"][Q], forecasts["adaptive_langgraph"][Q])
    assert decisions["adaptive_python"] == decisions["adaptive_langgraph"]
    assert results["adaptive_python"] == results["adaptive_langgraph"]
    levels = np.arange(1,100) / 100.0
    losses = {}
    for track in ("adaptive_python", "fixed_initial_counterfactual"):
        keyed = forecasts[track].merge(actual_frame, on="TIMESTAMP", how="inner", validate="one_to_one")
        error = keyed.net_mw.to_numpy()[:,None] - keyed[Q].to_numpy()
        losses[track] = np.maximum(levels * error, (levels - 1) * error).mean(axis=1)
    difference = losses["adaptive_python"] - losses["fixed_initial_counterfactual"]
    paired = {"contrast": "adaptive minus fixed initial hourly Q01-Q99 pinball; negative favors adaptive",
              "interpretation": "extra training effort/selection policy, not agentic executor superiority",
              "primary_hac_lag168": hac_mean(difference, 168),
              "sensitivity_hac_lag24": hac_mean(difference, 24),
              "exploratory_reused_test": True}
    report = {"experiment": "raw measured source to bounded training, fusion and research decisions",
        "domain_training_reports": reports, "joint": results,
        "matched_adaptive_forecasts_and_decisions_exact": True,
        "paired_extra_effort_comparison": paired,
        "fixed_initial_scope": "same first trial; extra training effort/policy counterfactual, not an agentic comparator",
        "autonomy_scope": "bounded deterministic training control; no LLM reasoning, human study or autonomous grid action",
        "test_scope": "reused 2019 chronological test, not independent new holdout",
        "timing_scope": "one execution per domain/executor; training domains ran concurrently; no runtime superiority inference",
        "file_sha256": {str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--scenarios", type=int, default=1000)
    args = parser.parse_args()
    result = run(args.base, scenarios=args.scenarios)
    print(json.dumps({"joint": result["joint"], "matched": result["matched_adaptive_forecasts_and_decisions_exact"],
                      "scope": result["autonomy_scope"]}, indent=2))

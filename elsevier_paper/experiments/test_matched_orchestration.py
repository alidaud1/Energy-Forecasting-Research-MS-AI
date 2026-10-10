import json

import numpy as np
import pandas as pd

from elsevier_paper.experiments.run_matched_orchestration import (
    Q, build_graph, cases, python_pipeline, run, verify_audit,
)


def inputs():
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    load = pd.DataFrame({"TIMESTAMP": times, **{q: [80+i, 90+i, 100+i] for i, q in enumerate(Q)}})
    solar = pd.DataFrame({"TIMESTAMP": times, **{q: [i/3, i/4, i/5] for i, q in enumerate(Q)}})
    return load, solar


def test_all_faults_have_matched_outcomes_and_audits():
    graph = build_graph()
    for name, payload in cases(*inputs()).items():
        state = {**payload, "case": name, "events": [], "capacity": 60, "scenarios": 32, "seed": 2026}
        left, right = python_pipeline(state), graph.invoke(state)
        verify_audit(left); verify_audit(right)
        assert left["events"] == right["events"]
        assert left.get("error") == right.get("error")
        if not left.get("error"):
            np.testing.assert_array_equal(left["fused"].forecast[Q], right["fused"].forecast[Q])
            assert left["decision"] == right["decision"]


def test_safe_recovery_does_not_invent_missing_values():
    scenarios = cases(*inputs())
    for name, expected_error, expected_recovery in (
        ("reversed_timestamps", False, 1), ("transient_fusion_failure", False, 1),
        ("missing_quantile", True, 0), ("duplicate_timestamp", True, 0),
        ("persistent_fusion_failure", True, 0),
    ):
        state = {**scenarios[name], "case": name, "events": [], "capacity": 60, "scenarios": 32, "seed": 2026}
        result = python_pipeline(state)
        assert bool(result.get("error")) == expected_error
        assert sum(e["action"] == "recovered" for e in result["events"]) == expected_recovery
        if name == "persistent_fusion_failure":
            assert sum(e["action"] == "retry" for e in result["events"]) == 1
            assert "decision" not in result and "fused" not in result


def test_measured_benchmark_exports_independent_attempt_audits(tmp_path):
    report = run(tmp_path, repeats=2, hours=3, scenarios=32)
    assert report["runs"] == 56
    assert report["audit_complete_runs"] == 56
    assert report["rejected_cases_per_engine"] == 9
    assert report["recovered_cases_per_engine"] == 2
    assert report["negative_control_accepted_both"]
    assert len(list((tmp_path / "audits").rglob("*.jsonl"))) == 56
    assert json.loads((tmp_path / "summary.json").read_text())["protocol"]["same_stage_functions"]
    assert (pd.read_csv(tmp_path / "timings.csv").seconds > 0).all()

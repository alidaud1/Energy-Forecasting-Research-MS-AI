import numpy as np
import pandas as pd
import json
import hashlib
from pathlib import Path
import pytest

from elsevier_paper.experiments.run_training_autonomy import actual_mw, policy, plain_execute


def test_solar_actual_conversion_does_not_mutate_pandas_backed_targets():
    table = pd.DataFrame({"POWER": [.1, .2], "capacity_lag25": [100, 200]})
    before = table.copy(deep=True)
    np.testing.assert_array_equal(actual_mw(table, "solar"), [10, 40])
    pd.testing.assert_frame_equal(table, before)
    np.testing.assert_array_equal(actual_mw(pd.DataFrame({"LOAD": [10, 20]}), "load"), [10, 20])


def test_gate_selects_or_expands_only_within_declared_budgets():
    assert policy([90], 100)["action"] == "select"
    assert policy([96], 100)["action"] == "retrain"
    assert policy([96], 100)["next_budget"] == 50
    final = policy([96, 80], 100)
    assert final["action"] == "select" and final["selected_index"] == 1 and final["gate_met"]


def test_exhaustion_keeps_better_initial_candidate_and_flags_missed_target():
    final = policy([100, 110], 100)
    assert final["selected_index"] == 0
    assert final["target_missed_after_budget"] and final["action"] == "select"
    assert final["next_budget"] is None
    assert policy([100, 100], 100)["selected_index"] == 0


@pytest.mark.parametrize("scores,reference", [([], 100), ([np.nan], 100), ([1,2,3], 100), ([1], -1)])
def test_invalid_policy_input_is_rejected(scores, reference):
    with pytest.raises(ValueError):
        policy(scores, reference)


def test_plain_executor_retrains_without_repeating_preparation():
    history = []
    def node(name, next_step):
        def call(state):
            history.append(name)
            return {**state, "current_step": next_step}
        return call
    count = [0]
    def review(state):
        history.append("human_review")
        count[0] += 1
        return {**state, "human_decision": "retrain" if count[0] == 1 else "approve"}
    nodes = {"orchestrator": lambda state: state,
             **{name: node(name, next_step) for name, next_step in
                zip(("dataset_validator","data","feature_eng","stats","training","evaluation"),
                    ("data","feature_eng","stats","training","human_review","done"))},
             "human_review": review}
    result = plain_execute(nodes, {"current_step": "dataset_validator"})
    assert result["current_step"] == "done"
    assert history == ["dataset_validator", "data", "feature_eng", "stats", "training",
                       "human_review", "training", "human_review", "evaluation"]


@pytest.mark.parametrize("domain", ["load", "solar"])
def test_saved_full_training_paths_and_checkpoint_provenance(domain):
    base = Path(__file__).resolve().parent / "training_autonomy" / domain
    result = json.loads((base / "summary.json").read_text())
    assert result["paired_forecasts_and_selection_exact"]
    assert hashlib.sha256((base / "runner_source.py").read_bytes()).hexdigest() == result["script_sha256"]
    for engine in ("python", "langgraph"):
        report = result["reports"][engine]
        assert report["test_hours"] == 8760 and report["test_used_for_selection"] is False
        assert report["scripted_human_approval"] is False
        assert report["completed_steps"].count("dataset_validator") == 1
        assert report["completed_steps"].count("training") == len(report["trials"]) <= 2
        expected = policy([t["validation"]["mean_pinball_loss_mw"] for t in report["trials"]],
                          report["reference_validation_pinball_mw"])
        assert report["selection"] == expected and expected["action"] == "select"
        events = [json.loads(line) for line in (base / engine / "events.jsonl").read_text().splitlines()]
        assert [e["sequence"] for e in events] == list(range(len(events)))
        assert len(events) == report["audit_events"]
        for trial in report["trials"]:
            config = json.loads((base / engine / ("trial_" + str(trial["budget"])) / "configuration.json").read_text())
            assert config["checkpoint_reload_verified"] and not config["test_input_passed_to_trainer"]
            assert hashlib.sha256(Path(config["checkpoint"]).read_bytes()).hexdigest() == config["checkpoint_sha256"]
            assert len(config["actual_boosting_rounds"]) == 99
            assert max(config["actual_boosting_rounds"]) <= trial["budget"]

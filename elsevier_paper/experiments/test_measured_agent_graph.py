"""Measured graph hand-off checks, without model training or artifact writes."""

import json
from pathlib import Path

import pytest

from elsevier_paper.experiments.run_measured_agent_graph import validate_handoff

INPUTS = Path(__file__).resolve().parent / "opsd_agent_port"
GRAPH = Path(__file__).resolve().parent / "opsd_measured_full_graph"


@pytest.mark.parametrize("domain", ["load", "solar"])
def test_archived_measured_handoff_passes(domain):
    manifest, tables = validate_handoff(INPUTS, domain)
    assert manifest["area"] == "DE"
    assert [len(tables[s]) for s in ("train", "calibration", "test")] == [
        12934, 4417, 8760]


def test_wrong_forecast_issue_is_rejected(tmp_path):
    manifest = json.loads((INPUTS / "input_manifest.json").read_text())
    manifest["forecast_issue"] = "at target time"
    (tmp_path / "input_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="forecast-issue"):
        validate_handoff(tmp_path, "load")


def test_tampered_split_is_rejected(tmp_path):
    manifest = json.loads((INPUTS / "input_manifest.json").read_text())
    (tmp_path / "input_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "load_train.csv").write_text("TIMESTAMP,AREA,LOAD\n")
    with pytest.raises(ValueError, match="archived hand-off"):
        validate_handoff(tmp_path, "load")


def test_saved_graph_scores_reproduce_native_port():
    native = json.loads((INPUTS.parent / "opsd_native_subset/evaluation.json").read_text())
    for domain in ("load", "solar"):
        graph = json.loads((GRAPH / domain / "graph_evaluation.json").read_text())
        assert graph["test_hours"] == native["test_hours"] == 8760
        assert graph["scripted_review_not_human"] is True
        assert graph["metrics"]["mean_pinball_loss_mw"] == pytest.approx(
            native[domain]["mean_pinball_loss_mw"], abs=1e-9)
    joint = json.loads((GRAPH / "joint/joint_evaluation.json").read_text())
    for method in ("copula", "independent"):
        assert joint["net_load"][method]["metrics"]["mean_pinball_loss_mw"] == pytest.approx(
            native["net_load"][method]["mean_pinball_loss_mw"], abs=1e-9)


def test_saved_graph_forecasts_align_exactly():
    import pandas as pd

    load = pd.read_csv(GRAPH / "load/test_q01_q99.csv", usecols=["TIMESTAMP"])
    solar = pd.read_csv(GRAPH / "solar/test_q01_q99.csv", usecols=["TIMESTAMP"])
    assert len(load) == len(solar) == 8760
    assert load.TIMESTAMP.equals(solar.TIMESTAMP)

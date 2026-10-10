"""Regression and tamper checks for measured decision-risk proxies."""

from pathlib import Path

import pandas as pd
import pytest

from elsevier_paper.experiments.evaluate_opsd_2020_decisions import _variant


def test_saved_decision_proxy_audit_recomputes(tmp_path):
    from elsevier_paper.experiments.evaluate_opsd_2020_decisions import evaluate

    root = Path(__file__).resolve().parent
    report = evaluate(root / "opsd_de_baseline_2020", root / "opsd_native_2020",
                      tmp_path / "audit.json")
    assert len(report["variants"]) == 6
    assert report["no_operator_reserve_or_action_ground_truth"] is True
    assert report["variants"]["baseline_copula"]["test_hours"] == 4367
    assert report["variants"]["baseline_copula"]["q05_lower_exceedance_fraction"] > 0.25


def test_decision_proxy_rejects_mismatched_timestamp():
    actual = pd.DataFrame({"TIMESTAMP": ["2020-01-01T00:00:00Z"],
                           "net_load_actual_mw": [1.0]})
    forecast = pd.DataFrame({"TIMESTAMP": ["2020-01-01T01:00:00Z"],
                             "q05": [0.0], "q50": [1.0], "q95": [2.0]})
    with pytest.raises(ValueError, match="timestamps differ"):
        _variant(actual, forecast, [])

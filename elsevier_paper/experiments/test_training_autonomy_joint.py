import numpy as np
import pandas as pd
import json
from pathlib import Path

from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS
from elsevier_paper.experiments.evaluate_training_autonomy import decision_node, fusion_node, joint_graph


def test_connected_fusion_graph_matches_plain_nodes_without_actuals():
    stamp = pd.date_range("2019-01-01", periods=4, freq="h")
    def bundle(source, center, bound=None):
        return ForecastBundle(source, pd.DataFrame({"TIMESTAMP": stamp,
            **{q: np.repeat(center+i/10, len(stamp)) for i,q in enumerate(QUANTILE_COLUMNS)}}),
            "MW", "DE", "24h", physical_min=0, physical_max=bound,
            metadata={"capacity_mw": bound} if bound else {})
    state = {"load": bundle("load", 100), "solar": bundle("solar", 20, 60),
             "dependence": -.25, "seed": 42, "scenarios": 100}
    plain = {**state, **fusion_node(state)}
    plain.update(decision_node(plain))
    graph = joint_graph().invoke(state)
    np.testing.assert_array_equal(plain["fused"].forecast[list(QUANTILE_COLUMNS)],
                                  graph["fused"].forecast[list(QUANTILE_COLUMNS)])
    assert plain["decision"] == graph["decision"]


def test_saved_end_to_end_joint_result_matches_executors_and_is_scoped():
    base = Path(__file__).resolve().parent / "training_autonomy/joint"
    result = json.loads((base / "summary.json").read_text())
    assert result["matched_adaptive_forecasts_and_decisions_exact"]
    left = pd.read_csv(base / "adaptive_python_net_q01_q99.csv")
    right = pd.read_csv(base / "adaptive_langgraph_net_q01_q99.csv")
    pd.testing.assert_frame_equal(left, right)
    assert len(left) == 8760 and left.shape[1] == 100
    assert np.isfinite(left[list(QUANTILE_COLUMNS)]).all().all()
    assert (np.diff(left[list(QUANTILE_COLUMNS)], axis=1) >= 0).all()
    assert result["joint"]["adaptive_python"] == result["joint"]["adaptive_langgraph"]
    decisions = json.loads((base / "adaptive_langgraph_decisions.json").read_text())
    assert len(decisions) == 12
    assert all("not autonomous grid dispatch" in item["scope"] for item in decisions)
    assert result["paired_extra_effort_comparison"]["exploratory_reused_test"]

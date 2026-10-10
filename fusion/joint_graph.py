"""LangGraph wrapper for the post-forecast joint workflow."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .adapters import load_bundle, solar_bundle
from .alignment import align_cyclically, describe_overlap
from .decision_agent import make_decisions
from .fusion_agent import fuse_forecasts


class JointState(TypedDict, total=False):
    load_predictions: str
    solar_predictions: str
    solar_timestamps: str
    solar_capacity_mw: float
    solar_location_value: object
    n_scenarios: int
    seed: int
    dependence: float | None
    fused_forecast: object
    _fused_result: object
    fusion_diagnostics: dict
    alignment: dict
    decision_summary: dict
    recommendations: list[str]
    risk_flags: list[str]
    errors: list[str]


def fusion_node(state: JointState) -> dict:
    """Load and solar artifact hand-off; numerical fusion is deterministic."""
    try:
        if "solar_capacity_mw" not in state:
            raise ValueError("solar_capacity_mw is required: solar forecasts are a "
                             "fraction of installed capacity, not MW")
        load = load_bundle(state["load_predictions"])
        solar = solar_bundle(
            state["solar_predictions"],
            timestamps_path=state["solar_timestamps"],
            capacity_mw=state["solar_capacity_mw"],
            location_value=state.get("solar_location_value", 1),
        )
        overlap = describe_overlap(load, solar)
        if overlap["requires_alignment"]:
            solar = align_cyclically(load, solar)
        fused = fuse_forecasts(
            load, solar, n_scenarios=state.get("n_scenarios", 1000),
            seed=state.get("seed", 42), dependence=state.get("dependence"),
        )
        return {"fused_forecast": fused.forecast,
                "fusion_diagnostics": fused.diagnostics,
                "alignment": {**overlap, **{k: v for k, v in solar.metadata.items()
                                            if k.startswith("alignment") or k in
                                            ("capacity_mw", "capacity_basis",
                                             "solar_local_offset_hours", "caveat")}},
                "_fused_result": fused}
    except Exception as exc:
        return {"errors": [f"fusion: {exc}"]}


def decision_node(state: JointState) -> dict:
    """Convert fused scenarios into auditable operational guidance."""
    if state.get("errors"):
        return {}
    fused = state.get("_fused_result")
    if fused is None:
        return {"errors": ["decision: fusion result is missing"]}
    try:
        decision = make_decisions(fused)
        return {"decision_summary": decision.summary,
                "recommendations": decision.recommendations,
                "risk_flags": decision.risk_flags}
    except Exception as exc:
        return {"errors": [f"decision: {exc}"]}


def build_joint_graph():
    graph = StateGraph(JointState)
    graph.add_node("fusion", fusion_node)
    graph.add_node("decision", decision_node)
    graph.set_entry_point("fusion")
    graph.add_edge("fusion", "decision")
    graph.add_edge("decision", END)
    return graph.compile()

"""Shared interfaces for joint load/solar forecasting."""

from .forecast_contract import ForecastBundle, ForecastContractError, QUANTILE_COLUMNS
from .adapters import load_bundle, solar_bundle
from .alignment import align_cyclically, describe_overlap, solar_local_offset
from .fusion_agent import FusionResult, fuse_forecasts
from .decision_agent import DecisionResult, make_decisions
from .joint_pipeline import run_joint_pipeline

__all__ = ["ForecastBundle", "ForecastContractError", "QUANTILE_COLUMNS",
           "load_bundle", "solar_bundle",
           "align_cyclically", "describe_overlap", "solar_local_offset",
           "FusionResult", "fuse_forecasts", "DecisionResult", "make_decisions",
           "run_joint_pipeline", "JointState", "build_joint_graph"]


def __getattr__(name):
    """Defer the LangGraph import so the contract works without it installed.

    Only joint_graph needs langgraph; importing it eagerly meant the validation
    layer and the tests could not be used in an environment that lacks it.
    """
    if name in ("JointState", "build_joint_graph"):
        from . import joint_graph
        return getattr(joint_graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

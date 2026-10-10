"""Reusable joint load/solar integration entry point."""

from __future__ import annotations

import json
from pathlib import Path

from .adapters import load_bundle, solar_bundle
from .alignment import align_cyclically, describe_overlap
from .decision_agent import make_decisions
from .fusion_agent import fuse_forecasts


def run_joint_pipeline(load_predictions: str | Path,
                       solar_predictions: str | Path,
                       solar_timestamps: str | Path,
                       output_dir: str | Path,
                       *, solar_capacity_mw: float,
                       solar_location_value: object = 1,
                       n_scenarios: int = 1000, seed: int = 42,
                       dependence: float | None = None,
                       solar_capacity_basis: str | None = None) -> dict:
    """Run adapters -> alignment -> fusion -> decisions and persist a report.

    The GEFCom load and solar test windows do not overlap, so alignment is
    applied whenever the calendars share no timestamps.  Whether it was applied,
    and under exactly which rule, is written into the manifest: the result is a
    methodological demonstration and the manifest is what says so.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    load = load_bundle(load_predictions)
    solar = solar_bundle(solar_predictions, timestamps_path=solar_timestamps,
                         capacity_mw=solar_capacity_mw,
                         location_value=solar_location_value,
                         location=f"solar-zone-{solar_location_value}",
                         capacity_basis=solar_capacity_basis)
    overlap = describe_overlap(load, solar)
    if overlap["requires_alignment"]:
        solar = align_cyclically(load, solar)
    fused = fuse_forecasts(load, solar, n_scenarios=n_scenarios, seed=seed,
                           dependence=dependence)
    decisions = make_decisions(fused)
    forecast_path = output / "net_load_forecast.csv"
    decision_path = output / "decision_report.json"
    manifest_path = output / "joint_manifest.json"
    fused.forecast.to_csv(forecast_path, index=False)
    decision_path.write_text(json.dumps({
        "summary": decisions.summary,
        "risk_flags": decisions.risk_flags,
        "recommendations": decisions.recommendations,
    }, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "load_predictions": str(Path(load_predictions).resolve()),
        "solar_predictions": str(Path(solar_predictions).resolve()),
        "solar_timestamps": str(Path(solar_timestamps).resolve()),
        "forecast_path": str(forecast_path),
        "decision_report_path": str(decision_path),
        "n_scenarios": n_scenarios,
        "seed": seed,
        "calendar_overlap": overlap,
        "assumptions": {
            "solar_capacity_mw": solar_capacity_mw,
            "solar_capacity_basis": solar.metadata.get("capacity_basis"),
            "alignment": solar.metadata.get("alignment", "none: calendars overlap"),
            "alignment_rule": solar.metadata.get("alignment_rule"),
            "solar_local_offset_hours": solar.metadata.get("solar_local_offset_hours"),
            "caveat": solar.metadata.get("caveat"),
        },
        "fusion_diagnostics": fused.diagnostics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return {"forecast_path": str(forecast_path),
            "decision_report_path": str(decision_path),
            "manifest_path": str(manifest_path),
            "decisions": decisions,
            "fusion": fused}

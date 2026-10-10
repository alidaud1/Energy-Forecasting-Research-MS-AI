"""CLI runner connecting the existing load and solar outputs to joint graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .joint_graph import build_joint_graph

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOAD = ROOT / "agentic/outputs/test_predictions_monotone/GBR_predictions_monotone.csv"
SOLAR_ROOT = ROOT / "agentic_solar/solar-forecasting-GEFCom2014"
DEFAULT_SOLAR = SOLAR_ROOT / "pipeline/outputs/predictions/Ensemble_predictions.csv"
BENCHMARK_INDEX = SOLAR_ROOT / "data/GEFCom2014-S_V2/Solar/Task 15/benchmark15.csv"
DEFAULT_OUTPUT = ROOT / "joint_outputs"

# GEFCom2014-S reports output as a fraction of installed capacity and never
# states that capacity.  60 MW is an ASSUMPTION, chosen as roughly 19% of the
# load track's observed peak (315.6 MW) so the array represents a realistic
# utility-scale penetration.  It is recorded in the manifest and every result
# scales linearly with it, so it must be reported alongside any figure.
DEFAULT_SOLAR_CAPACITY_MW = 60.0


def resolve_solar_index(solar_predictions: Path) -> Path:
    """Prefer the solar pipeline's own split for the model being fused.

    The split file carries the POWER actuals, which is what lets fusion estimate
    load/solar dependence instead of assuming independence.  The published
    benchmark file is the fallback; it indexes the same rows but has no actuals.
    Row counts are verified downstream, so a wrong guess fails loudly.
    """
    model = solar_predictions.name.replace("_predictions.csv", "")
    candidate = SOLAR_ROOT / f"pipeline/outputs/splits/{model}/test.csv"
    if candidate.is_file():
        return candidate
    try:
        rows = sum(1 for _ in solar_predictions.open()) - 1
    except OSError:
        return BENCHMARK_INDEX
    for split in sorted((SOLAR_ROOT / "pipeline/outputs/splits").glob("*/test.csv")):
        if sum(1 for _ in split.open()) - 1 == rows:
            return split
    return BENCHMARK_INDEX


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run joint load/solar net-load forecasting")
    parser.add_argument("--load", type=Path, default=DEFAULT_LOAD)
    parser.add_argument("--solar", type=Path, default=DEFAULT_SOLAR)
    parser.add_argument("--solar-timestamps", type=Path, default=None,
                        help="index artifact for the solar predictions "
                             "(default: the matching pipeline split, else benchmark15)")
    parser.add_argument("--solar-capacity-mw", type=float, default=DEFAULT_SOLAR_CAPACITY_MW,
                        help=f"ASSUMED installed capacity in MW (default {DEFAULT_SOLAR_CAPACITY_MW}); "
                             "GEFCom2014-S publishes only capacity fractions")
    parser.add_argument("--dependence", type=float, default=None,
                        help="stated Spearman load/solar residual correlation; "
                             "omit to estimate it from actuals where available")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zone", default=1, type=int)
    parser.add_argument("--scenarios", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args(argv)

    index = args.solar_timestamps or resolve_solar_index(args.solar.resolve())
    state = build_joint_graph().invoke({
        "load_predictions": str(args.load.resolve()),
        "solar_predictions": str(args.solar.resolve()),
        "solar_timestamps": str(Path(index).resolve()),
        "solar_capacity_mw": args.solar_capacity_mw,
        "solar_location_value": args.zone,
        "n_scenarios": args.scenarios,
        "seed": args.seed,
        "dependence": args.dependence,
    })
    if state.get("errors"):
        parser.error("; ".join(state["errors"]))
    args.output.mkdir(parents=True, exist_ok=True)
    state["fused_forecast"].to_csv(args.output / "net_load_forecast.csv", index=False)
    (args.output / "decision_report.json").write_text(json.dumps({
        "summary": state["decision_summary"],
        "risk_flags": state.get("risk_flags", []),
        "recommendations": state.get("recommendations", []),
    }, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output / "joint_manifest.json").write_text(json.dumps({
        "load_predictions": str(args.load.resolve()),
        "solar_predictions": str(args.solar.resolve()),
        "solar_index": str(Path(index).resolve()),
        "zone": args.zone, "scenarios": args.scenarios, "seed": args.seed,
        "assumptions": state.get("alignment", {}),
        "fusion_diagnostics": state["fusion_diagnostics"],
    }, indent=2, default=str) + "\n", encoding="utf-8")
    diag = state["fusion_diagnostics"]
    print(f"Joint report written to {args.output.resolve()}")
    print(f"  steps={diag['n_timestamps']}  scenarios={diag['n_scenarios']}  "
          f"capacity={diag['solar_capacity_mw']} MW")
    print(f"  alignment: {diag.get('alignment_rule') or diag.get('alignment')}")
    print(f"  dependence: {diag['dependence_source']} (rank {diag['rank_correlation']:.4f})")
    print(f"  flags: {state.get('risk_flags') or ['none']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

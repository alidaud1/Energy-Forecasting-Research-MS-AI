"""Regenerate the dependence sensitivity quoted in Section 5.3 of the thesis.

Section 5.3 argues that reading net-load quantiles off a scenario ensemble is
not interchangeable with subtracting the two forecasts' matching quantiles.
That argument rests on a number, so the number is produced by a script rather
than by hand: this module rebuilds the same two bundles the joint run uses,
fuses them at a range of stated rank correlations, and records the mean width of
the central 90 per cent net-load interval in each case.

The comparison case, ``same_rank_subtraction``, is the shortcut itself --
NetLoad_tau = Load_tau - Solar_tau at every quantile level.  Its interval width
is the load width minus the solar width, because the two spreads cancel term by
term, which is exactly the behaviour that makes the shortcut a perfect-
dependence assumption rather than a neutral simplification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .adapters import load_bundle, solar_bundle
from .alignment import align_cyclically, describe_overlap
from .forecast_contract import ForecastBundle
from .fusion_agent import fuse_forecasts
from .run_existing import (DEFAULT_LOAD, DEFAULT_OUTPUT, DEFAULT_SOLAR,
                           DEFAULT_SOLAR_CAPACITY_MW, resolve_solar_index)

# The reported band. q95 - q05 is the interval the reserve discussion uses, so
# the sensitivity is stated on the same band rather than on a second one.
LOW, HIGH = "q05", "q95"

# Stated Spearman correlations to sweep. 0.0 is the honest default for two
# error processes driven by demand behaviour and cloud cover respectively; the
# rest show how far the answer moves if that assumption is wrong.
RANK_CORRELATIONS = (0.0, 0.3, 0.6, 0.9)

# Assumed installed capacities for the penetration sweep, in MW. The shortcut's
# error is not a fixed quantity: it grows with the share of the net load the
# solar array carries, because the shortcut subtracts the solar spread from the
# load spread instead of combining them.
PENETRATION_CAPACITIES_MW = (60.0, 120.0, 180.0, 240.0, 300.0)

# Seeds used to measure the sampling error the method itself introduces. The
# claim that Monte Carlo error is negligible beside the shortcut's error is only
# worth making if it is measured, so the run repeats the fusion under different
# random draws and reports the spread of the answer.
CONVERGENCE_SEEDS = (101, 202, 303, 404)


def _mean_width(frame) -> float:
    return float((frame[HIGH] - frame[LOW]).mean())


def same_rank_width(load: ForecastBundle, solar: ForecastBundle) -> float:
    """Mean 90 per cent width the same-rank subtraction shortcut would report.

    Subtracting matching quantiles gives an interval whose width is the load
    width less the solar width: the two uncertainties cancel instead of
    compounding, which is only correct if the two series move in lockstep at
    every rank.
    """
    left = load.frame[["TIMESTAMP", LOW, HIGH]]
    right = solar.frame[["TIMESTAMP", LOW, HIGH]]
    merged = left.merge(right, on="TIMESTAMP", how="inner", suffixes=("_l", "_s"))
    low = merged[f"{LOW}_l"] - merged[f"{LOW}_s"]
    high = merged[f"{HIGH}_l"] - merged[f"{HIGH}_s"]
    return float((high - low).mean())


def build_bundles(load_path: Path, solar_path: Path, index_path: Path,
                  capacity_mw: float, zone: int) -> tuple[ForecastBundle, ForecastBundle]:
    load = load_bundle(load_path)
    solar = solar_bundle(solar_path, timestamps_path=index_path,
                         capacity_mw=capacity_mw, location_value=zone)
    if describe_overlap(load, solar)["requires_alignment"]:
        solar = align_cyclically(load, solar)
    return load, solar


def sweep(load: ForecastBundle, solar: ForecastBundle, *,
          n_scenarios: int, seed: int) -> dict:
    """Mean 90 per cent net-load width at each stated dependence, plus shortcut."""
    results = {}
    for rho in RANK_CORRELATIONS:
        fused = fuse_forecasts(load, solar, n_scenarios=n_scenarios, seed=seed,
                               dependence=rho)
        results[f"rho_{rho}"] = round(_mean_width(fused.forecast), 2)
    results["same_rank_subtraction"] = round(same_rank_width(load, solar), 2)

    independent = results["rho_0.0"]
    understatement = 100.0 * (independent - results["same_rank_subtraction"]) / independent
    results["understatement_pct"] = round(understatement, 2)

    results["load_width_mw"] = round(_mean_width(load.frame), 2)
    results["solar_width_mw"] = round(_mean_width(solar.frame), 2)
    return results


def convergence(load: ForecastBundle, solar: ForecastBundle, *,
                n_scenarios: int) -> dict:
    """Spread of the reported interval across independent random draws.

    Sampling is the one approximation the fusion method introduces.  Repeating
    it under different seeds and reporting the standard deviation of the answer
    puts a number on that approximation, which is what lets Section 4.7.1 compare
    it against the error the same-rank shortcut introduces instead.
    """
    widths = [_mean_width(fuse_forecasts(load, solar, n_scenarios=n_scenarios,
                                         seed=seed, dependence=0.0).forecast)
              for seed in CONVERGENCE_SEEDS]
    return {"n_scenarios": n_scenarios,
            "seeds": list(CONVERGENCE_SEEDS),
            "mean_width_mw": round(float(np.mean(widths)), 3),
            "std_width_mw": round(float(np.std(widths, ddof=1)), 3)}


def penetration_sweep(load_path: Path, solar_path: Path, index_path: Path,
                      zone: int, *, n_scenarios: int, seed: int) -> list[dict]:
    """Repeat the comparison at a range of assumed installed capacities.

    Reported net-load quantities scale with the assumed capacity, so the size of
    the shortcut's error is not a property of the method alone; it is a property
    of how much of the net load the solar array carries.  Sweeping the capacity
    separates the two.
    """
    rows = []
    for capacity in PENETRATION_CAPACITIES_MW:
        load, solar = build_bundles(load_path, solar_path, index_path, capacity, zone)
        fused = fuse_forecasts(load, solar, n_scenarios=n_scenarios, seed=seed,
                               dependence=0.0)
        sampled = _mean_width(fused.forecast)
        shortcut = same_rank_width(load, solar)
        rows.append({
            "capacity_mw": capacity,
            "sampled_mw": round(sampled, 2),
            "shortcut_mw": round(shortcut, 2),
            "understatement_pct": round(100.0 * (sampled - shortcut) / sampled, 2),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load", type=Path, default=DEFAULT_LOAD)
    parser.add_argument("--solar", type=Path, default=DEFAULT_SOLAR)
    parser.add_argument("--solar-timestamps", type=Path, default=None)
    parser.add_argument("--solar-capacity-mw", type=float,
                        default=DEFAULT_SOLAR_CAPACITY_MW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zone", default=1, type=int)
    parser.add_argument("--scenarios", default=2000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args(argv)

    index = args.solar_timestamps or resolve_solar_index(args.solar.resolve())
    load, solar = build_bundles(args.load.resolve(), args.solar.resolve(),
                                Path(index).resolve(), args.solar_capacity_mw,
                                args.zone)
    results = sweep(load, solar, n_scenarios=args.scenarios, seed=args.seed)
    results["band"] = f"{HIGH} - {LOW}"
    results["capacity_mw"] = args.solar_capacity_mw
    results["n_scenarios"] = args.scenarios
    results["seed"] = args.seed
    results["convergence"] = convergence(load, solar, n_scenarios=args.scenarios)
    results["penetration"] = penetration_sweep(
        args.load.resolve(), args.solar.resolve(), Path(index).resolve(),
        args.zone, n_scenarios=args.scenarios, seed=args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "dependence_sensitivity.json"
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Dependence sensitivity written to {destination}")
    print(f"  band: mean {HIGH} - {LOW}, {args.scenarios:,} scenarios, seed {args.seed}")
    for rho in RANK_CORRELATIONS:
        print(f"  stated rank correlation {rho:.1f}: {results[f'rho_{rho}']:.2f} MW")
    print(f"  same-rank subtraction:       {results['same_rank_subtraction']:.2f} MW "
          f"({results['understatement_pct']:.2f}% narrower than independent sampling)")
    print(f"  marginal widths: load {results['load_width_mw']:.2f} MW, "
          f"solar {results['solar_width_mw']:.2f} MW")
    c = results["convergence"]
    print(f"  sampling error over {len(c['seeds'])} seeds: "
          f"sd {c['std_width_mw']:.3f} MW on a {c['mean_width_mw']:.2f} MW width")
    print("  penetration sweep (sampled vs shortcut):")
    for row in results["penetration"]:
        print(f"    {row['capacity_mw']:5.0f} MW capacity: {row['sampled_mw']:7.2f} vs "
              f"{row['shortcut_mw']:7.2f} MW  ({row['understatement_pct']:5.2f}% understated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

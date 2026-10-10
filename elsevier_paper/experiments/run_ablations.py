"""Reproduce journal fusion ablations on the temporally held-out synthetic week.

Run from the repository root with the load environment:

    agentic/.venv/bin/python -m elsevier_paper.experiments.run_ablations

This script reads the frozen benchmark artifacts. It never re-estimates
dependence from evaluation-period actuals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from fusion.adapters import load_bundle, solar_bundle
from fusion.forecast_contract import QUANTILE_COLUMNS
from fusion.fusion_agent import fuse_forecasts
from .figure_style import FIGURE_WIDTH_IN, journal_style, save_figure


LEVELS = np.arange(1, 100, dtype=float) / 100.0


def _scores(frame: pd.DataFrame, actual: np.ndarray) -> dict:
    values = frame.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float)
    if values.shape[0] != len(actual):
        raise ValueError("forecast and evaluation actuals have different lengths")
    error = actual[:, None] - values
    loss = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1) * error)
    return {
        "mean_pinball_loss_mw": float(loss.mean()),
        "coverage_q05_q95": float(np.mean((actual >= values[:, 4]) & (actual <= values[:, 94]))),
        "mean_width_q05_q95_mw": float(np.mean(values[:, 94] - values[:, 4])),
        "median_mae_mw": float(np.mean(np.abs(actual - values[:, 49]))),
        "quantile_crossing_pairs": int(np.sum(np.diff(values, axis=1) < -1e-10)),
    }


def _same_rank(load: pd.DataFrame, solar: pd.DataFrame) -> pd.DataFrame:
    left = load.loc[:, ["TIMESTAMP", *QUANTILE_COLUMNS]].copy()
    right = solar.loc[:, ["TIMESTAMP", *QUANTILE_COLUMNS]].copy()
    if not left["TIMESTAMP"].reset_index(drop=True).equals(
            right["TIMESTAMP"].reset_index(drop=True)):
        raise ValueError("same-rank ablation requires identical timestamps")
    out = pd.DataFrame({"TIMESTAMP": left["TIMESTAMP"].to_numpy()})
    out.loc[:, QUANTILE_COLUMNS] = (
        left.loc[:, QUANTILE_COLUMNS].to_numpy(float)
        - right.loc[:, QUANTILE_COLUMNS].to_numpy(float)
    )
    return out


@journal_style()
def _figures(root: Path, actuals: pd.DataFrame, frames: dict[str, pd.DataFrame],
             metrics: dict[str, dict]) -> list[str]:
    out = root / "figures"
    out.mkdir(exist_ok=True)
    written = []
    evaluation = actuals.iloc[-len(frames["copula_bounded"]):]
    dates = pd.to_datetime(evaluation["TIMESTAMP"])
    baseline = frames["copula_bounded"]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.0))
    ax.fill_between(dates, baseline["q05"].to_numpy(float),
                    baseline["q95"].to_numpy(float), color="#69a3d0", alpha=0.35,
                    label="Q05–Q95")
    ax.plot(dates, baseline["q50"], color="#075985", lw=1.5, label="Q50")
    ax.plot(dates, evaluation["net_load_actual_MW"], color="#262626", lw=0.9,
            label="Synthetic actual")
    ax.set(ylabel="Net load (MW)", xlabel="Held-out date, January 2020")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, columnspacing=1.0, handlelength=1.6,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    path = out / "heldout_net_load_fan.pdf"
    save_figure(fig, path)
    plt.close(fig)
    written.append(str(path))

    names = ["same_rank", "independent_bounded", "copula_bounded", "copula_unbounded"]
    labels = ["Same-rank\nsubtraction", "Independent\nsampling",
              "Copula +\nbounds", "Copula,\nunbounded"]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(FIGURE_WIDTH_IN, 6.2))
    for ax, key, title, ylabel in zip(
        axes,
        ("mean_pinball_loss_mw", "coverage_q05_q95", "mean_width_q05_q95_mw"),
        ("Mean pinball loss", "Q05–Q95 coverage", "Mean Q05–Q95 width"),
        ("MW", "Fraction", "MW"),
    ):
        vals = [metrics[n][key] for n in names]
        ax.bar(positions, vals, color=["#9ca3af", "#60a5fa", "#0f766e", "#e5a64c"])
        ax.set_xticks(positions, labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        if key == "coverage_q05_q95":
            ax.axhline(0.9, color="#b91c1c", ls="--", lw=1, label="Nominal 0.90")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="upper left", frameon=True, facecolor="white",
                      edgecolor="none", framealpha=1)
    fig.tight_layout(h_pad=1.2)
    path = out / "fusion_ablation.pdf"
    save_figure(fig, path)
    plt.close(fig)
    written.append(str(path))
    return written


def run(root: str | Path, *, n_scenarios: int | None = None,
        make_figures: bool = True) -> dict:
    root = Path(root).resolve()
    benchmark = json.loads((root / "joint_evaluation.json").read_text())
    if benchmark.get("dependence_uses_evaluation_actuals") is not False:
        raise ValueError("ablation requires a leakage-free temporal holdout")
    if n_scenarios is None:
        n_scenarios = int(benchmark["n_scenarios"])
    seed = int(benchmark["seed"])
    rho = float(benchmark["calibration_rank_correlation"])
    capacity = float(benchmark["solar_capacity_mw"])
    load = load_bundle(root / "load_predictions_test.csv")
    solar = solar_bundle(root / "solar_predictions_test.csv",
                         timestamps_path=root / "solar_index_test.csv",
                         capacity_mw=capacity, location_value=1,
                         capacity_basis="ASSUMED: installed capacity of simulated grid")
    actuals = pd.read_csv(root / "synthetic_actuals.csv")
    evaluation = actuals.iloc[-len(load.frame):].reset_index(drop=True)
    expected = pd.to_datetime(evaluation["TIMESTAMP"]).reset_index(drop=True)
    if not expected.equals(load.frame["TIMESTAMP"].reset_index(drop=True)):
        raise ValueError("held-out actuals do not align with forecasts")
    actual = evaluation["net_load_actual_MW"].to_numpy(float)

    independent = fuse_forecasts(load, solar, dependence=0.0,
                                 n_scenarios=n_scenarios, seed=seed)
    copula = fuse_forecasts(load, solar, dependence=rho,
                            n_scenarios=n_scenarios, seed=seed)
    no_bounds = fuse_forecasts(replace(load, physical_min=None),
                               replace(solar, physical_min=None, physical_max=None),
                               dependence=rho, n_scenarios=n_scenarios, seed=seed)
    frames = {
        "same_rank": _same_rank(load.frame, solar.frame),
        "independent_bounded": independent.forecast,
        "copula_bounded": copula.forecast,
        "copula_unbounded": no_bounds.forecast,
    }
    if n_scenarios == int(benchmark["n_scenarios"]):
        stored = pd.read_csv(root / "joint_run/net_load_forecast.csv")
        np.testing.assert_allclose(stored.loc[:, QUANTILE_COLUMNS],
                                   copula.forecast.loc[:, QUANTILE_COLUMNS], rtol=0, atol=1e-10)

    metrics = {name: _scores(frame, actual) for name, frame in frames.items()}
    metrics["copula_unbounded"].update({
        "solar_samples_below_zero": int(np.sum(no_bounds.solar_scenarios < 0)),
        "solar_samples_above_capacity": int(np.sum(no_bounds.solar_scenarios > capacity)),
        "load_samples_below_zero": int(np.sum(no_bounds.load_scenarios < 0)),
    })
    result = {
        "benchmark": "controlled_synthetic_temporal_holdout",
        "not_measured_utility_data": True,
        "evaluation_start": benchmark["evaluation_start"],
        "evaluation_end": benchmark["evaluation_end"],
        "evaluation_hours": len(actual),
        "calibration_rank_correlation": rho,
        "seed": seed,
        "n_scenarios": n_scenarios,
        "methods": metrics,
        "same_rank_note": "Raw Q_L(tau)-Q_S(tau); may cross and is not a valid joint quantile model.",
        "physical_bounds_note": "No-bounds ablation removes scenario clipping, not the generator's quantile clipping.",
    }
    (root / "ablation_results.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame.from_dict(metrics, orient="index").rename_axis("method").to_csv(
        root / "ablation_results.csv")
    if make_figures:
        result["figures"] = _figures(root, actuals, frames, metrics)
        (root / "ablation_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("elsevier_paper/experiments/synthetic_holdout"))
    parser.add_argument("--scenarios", type=int, default=None)
    args = parser.parse_args()
    result = run(args.input, n_scenarios=args.scenarios)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

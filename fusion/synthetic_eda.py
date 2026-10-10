"""Generate EDA plots for the synchronized synthetic joint benchmark.

Usage from the repository root::

    agentic/.venv/bin/python -m fusion.synthetic_eda \
      --input joint_outputs/synthetic_benchmark

Plots are diagnostic only; this script does not alter forecasts or scenarios.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QCOLS = [f"q{i:02d}" for i in range(1, 100)]


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_eda(input_dir: str | Path, output_dir: str | Path | None = None) -> dict:
    root = Path(input_dir).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "eda_outputs"
    out.mkdir(parents=True, exist_ok=True)
    actual = pd.read_csv(root / "synthetic_actuals.csv", parse_dates=["TIMESTAMP"])
    load = pd.read_csv(root / "load_predictions.csv", parse_dates=["TIMESTAMP"])
    solar = pd.read_csv(root / "solar_predictions.csv", parse_dates=["TIMESTAMP"])
    fused_path = root / "joint_run" / "net_load_forecast.csv"
    fused = pd.read_csv(fused_path, parse_dates=["TIMESTAMP"]) if fused_path.is_file() else None

    # 1. Actual synchronized series and net load.
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(actual.TIMESTAMP, actual.load_actual_MW, label="Load actual (MW)", lw=1.2)
    ax.plot(actual.TIMESTAMP, actual.solar_actual_MW, label="Solar actual (MW)", lw=1.2)
    ax.plot(actual.TIMESTAMP, actual.net_load_actual_MW, label="Net load actual (MW)", lw=1.5)
    ax.set(title="Synchronized synthetic load, solar and net load", xlabel="Timestamp", ylabel="MW")
    ax.legend(ncol=3); ax.grid(alpha=.25)
    _save(fig, out / "01_synchronized_actuals.png")

    # 2. Distribution comparison.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    series = [(actual.load_actual_MW, "Load", "#1f77b4"),
              (actual.solar_actual_MW, "Solar", "#f2a900"),
              (actual.net_load_actual_MW, "Net load", "#2ca02c")]
    for ax, (values, title, color) in zip(axes, series):
        ax.hist(values, bins=30, color=color, alpha=.8, edgecolor="white")
        ax.axvline(np.median(values), color="black", ls="--", label=f"Median {np.median(values):.1f}")
        ax.set(title=f"{title} distribution", xlabel="MW", ylabel="Count")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    _save(fig, out / "02_distributions.png")

    # 3. Typical daily profiles by scenario.
    tmp = actual.assign(hour=actual.TIMESTAMP.dt.hour)
    grouped = tmp.groupby(["scenario", "hour"], observed=True)[
        ["load_actual_MW", "solar_actual_MW", "net_load_actual_MW"]].mean()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for col, ax, title in zip(["load_actual_MW", "solar_actual_MW", "net_load_actual_MW"], axes,
                               ["Load profile", "Solar profile", "Net-load profile"]):
        for scenario, values in grouped[col].groupby(level=0):
            ax.plot(values.index.get_level_values("hour"), values.to_numpy(), label=scenario)
        ax.set_ylabel("MW"); ax.set_title(title); ax.grid(alpha=.2)
    axes[-1].set_xlabel("Hour of day"); axes[0].legend(ncol=3, fontsize=8)
    _save(fig, out / "03_daily_profiles_by_scenario.png")

    # 4. Scenario distributions (manual boxplots avoid seaborn dependency).
    scenarios = list(dict.fromkeys(actual.scenario.tolist()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, title in zip(axes, ["load_actual_MW", "solar_actual_MW", "net_load_actual_MW"],
                              ["Load by scenario", "Solar by scenario", "Net load by scenario"]):
        ax.boxplot([actual.loc[actual.scenario == s, col] for s in scenarios], tick_labels=scenarios,
                   showfliers=False)
        ax.set_title(title); ax.set_ylabel("MW"); ax.tick_params(axis="x", rotation=35); ax.grid(alpha=.2)
    _save(fig, out / "04_scenario_boxplots.png")

    # 5. Actual load/solar dependence and net-load relationship.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(actual.load_actual_MW, actual.solar_actual_MW, c=actual.TIMESTAMP.dt.hour,
                    cmap="viridis", s=12, alpha=.7)
    axes[0].set(xlabel="Load (MW)", ylabel="Solar (MW)", title="Load–solar dependence")
    axes[0].grid(alpha=.2)
    axes[1].scatter(actual.solar_actual_MW, actual.net_load_actual_MW, c=actual.TIMESTAMP.dt.hour,
                    cmap="plasma", s=12, alpha=.7)
    axes[1].set(xlabel="Solar (MW)", ylabel="Net load (MW)", title="Solar vs net load")
    axes[1].grid(alpha=.2)
    _save(fig, out / "05_dependence_scatter.png")

    # 6. Forecast fan charts and actuals.
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for frame, ax, actual_col, title, color in [(load, axes[0], "load_actual_MW", "Load forecast quantiles", "#1f77b4"),
                                                  (solar, axes[1], "solar_actual_MW", "Solar forecast quantiles (MW)", "#f2a900")]:
        actual_values = actual[actual_col].to_numpy()
        scaled = frame[QCOLS].to_numpy(float)
        if frame is solar:
            # Solar forecasts are capacity fractions; the generated benchmark
            # manifest records 60 MW as its installed-capacity assumption.
            scaled = scaled * float(actual.solar_actual_MW.max() / max(actual.solar_actual_fraction.max(), 1e-9))
        ax.fill_between(frame.TIMESTAMP, scaled[:, 4], scaled[:, 94], color=color, alpha=.2, label="Q05–Q95")
        ax.plot(frame.TIMESTAMP, scaled[:, 49], color=color, label="Q50")
        ax.plot(actual.TIMESTAMP, actual_values, color="black", lw=.8, label="Actual")
        ax.set_title(title); ax.set_ylabel("MW"); ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.2)
    _save(fig, out / "06_forecast_fan_charts.png")

    # 7. Fused net-load fan chart (if the joint run has been executed).
    if fused is not None:
        fig, ax = plt.subplots(figsize=(14, 5))
        fq = fused[QCOLS].to_numpy(float)
        ax.fill_between(fused.TIMESTAMP, fq[:, 4], fq[:, 94], color="#2ca02c", alpha=.22, label="Q05–Q95")
        ax.plot(fused.TIMESTAMP, fq[:, 49], color="#2ca02c", label="Fused Q50")
        ax.plot(actual.TIMESTAMP, actual.net_load_actual_MW, color="black", lw=.9, label="Actual net load")
        ax.set(title="Fusion Agent net-load forecast", xlabel="Timestamp", ylabel="MW")
        ax.legend(ncol=3); ax.grid(alpha=.25)
        _save(fig, out / "07_fused_net_load_fan.png")

    summary = {
        "input": str(root), "output": str(out), "plots": sorted(p.name for p in out.glob("*.png")),
        "rows": int(len(actual)), "start": str(actual.TIMESTAMP.min()), "end": str(actual.TIMESTAMP.max()),
        "scenarios": sorted(actual.scenario.unique().tolist()),
        "missing_values": int(actual.isna().sum().sum()),
        "load_min_max_mw": [float(actual.load_actual_MW.min()), float(actual.load_actual_MW.max())],
        "solar_min_max_mw": [float(actual.solar_actual_MW.min()), float(actual.solar_actual_MW.max())],
        "net_load_min_max_mw": [float(actual.net_load_actual_MW.min()), float(actual.net_load_actual_MW.max())],
        "load_solar_pearson": float(actual.load_actual_MW.corr(actual.solar_actual_MW)),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("joint_outputs/synthetic_benchmark"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = generate_eda(args.input, args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

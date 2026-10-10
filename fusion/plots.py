"""Figures for the net-load fusion results.

Written by the fusion module rather than the thesis build, so every plot comes
from the same code path that produced the numbers beside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adapters import load_bundle, solar_bundle
from .alignment import align_cyclically
from .decision_agent import make_decisions
from .fusion_agent import fuse_forecasts

DPI = 400
LOAD_C, SOLAR_C, NET_C = "#1f4e79", "#e8a33d", "#2e7d32"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def duck_curve(load, solar, result, out: Path) -> Path:
    """Mean diurnal demand against mean diurnal net load."""
    hours = pd.DatetimeIndex(result.forecast["TIMESTAMP"]).hour
    load_median = np.median(result.load_scenarios, axis=0)
    solar_median = np.median(result.solar_scenarios, axis=0)
    net_median = result.forecast["q50"].to_numpy()
    frame = pd.DataFrame({"hour": hours, "load": load_median,
                          "solar": solar_median, "net": net_median})
    mean = frame.groupby("hour").mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(mean.index, mean["load"], color=LOAD_C, lw=2.0, label="Demand (Q50)")
    ax.plot(mean.index, mean["net"], color=NET_C, lw=2.0, label="Net load (Q50)")
    ax.fill_between(mean.index, mean["net"], mean["load"], color=SOLAR_C,
                    alpha=0.45, label="Solar generation")
    peak, trough = int(mean["net"].idxmax()), int(mean["net"].idxmin())
    low, high = mean["net"].min(), mean["load"].max()
    span = high - low
    ax.set_ylim(low - 0.42 * span, high + 0.22 * span)
    ax.annotate(f"net peak {mean['net'].max():.0f} MW at {peak:02d}:00",
                xy=(peak, mean["net"].max()),
                xytext=(peak - 9.0, high + 0.13 * span), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
    ax.annotate(f"net trough {mean['net'].min():.0f} MW at {trough:02d}:00",
                xy=(trough, mean["net"].min()),
                xytext=(trough - 4.2, low - 0.20 * span), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    _style(ax, "Mean diurnal demand and net load", "Hour of day", "Power (MW)")
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def fan_chart(result, out: Path, days: int = 7) -> Path:
    """Net-load predictive intervals over a representative week."""
    frame = result.forecast.iloc[:days * 24]
    stamps = pd.DatetimeIndex(frame["TIMESTAMP"])
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for lo, hi, alpha in (("q05", "q95", 0.18), ("q25", "q75", 0.30)):
        ax.fill_between(x, frame[lo], frame[hi], color=NET_C, alpha=alpha,
                        linewidth=0, label=f"{lo.upper()}–{hi.upper()}")
    ax.plot(x, frame["q50"], color=NET_C, lw=1.4, label="Q50")
    ticks = np.arange(0, len(frame), 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([stamps[t].strftime("%d %b") for t in ticks], fontsize=8)
    ax.set_xlim(0, len(frame) - 1)
    _style(ax, f"Net-load predictive intervals, first {days} days",
           "Date", "Net load (MW)")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    from .run_existing import (DEFAULT_LOAD, DEFAULT_SOLAR, DEFAULT_SOLAR_CAPACITY_MW,
                              resolve_solar_index)
    parser = argparse.ArgumentParser(description="Plot the net-load fusion results")
    parser.add_argument("--load", type=Path, default=DEFAULT_LOAD)
    parser.add_argument("--solar", type=Path, default=DEFAULT_SOLAR)
    parser.add_argument("--solar-capacity-mw", type=float, default=DEFAULT_SOLAR_CAPACITY_MW)
    parser.add_argument("--zone", type=int, default=1)
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "joint_outputs" / "plots")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    load = load_bundle(args.load)
    solar = solar_bundle(args.solar, timestamps_path=resolve_solar_index(args.solar.resolve()),
                         capacity_mw=args.solar_capacity_mw, location_value=args.zone)
    aligned = align_cyclically(load, solar)
    result = fuse_forecasts(load, aligned, n_scenarios=args.scenarios, seed=args.seed)
    written = [
        duck_curve(load, aligned, result, args.output / "01_net_load_duck_curve.png"),
        fan_chart(result, args.output / "02_net_load_fan_chart.png"),
    ]
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

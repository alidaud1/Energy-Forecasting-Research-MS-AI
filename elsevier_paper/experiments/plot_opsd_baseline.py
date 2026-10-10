"""Plot the saved measured OPSD test forecasts without refitting any model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .figure_style import FIGURE_WIDTH_IN, journal_style, save_figure

@journal_style()
def plot(root: Path) -> Path:
    root = root.resolve()
    actual = pd.read_csv(root / "actuals_2019.csv", parse_dates=["TIMESTAMP"])
    copula = pd.read_csv(root / "net_load_copula_q01_q99_2019.csv",
                         parse_dates=["TIMESTAMP"])
    independent = pd.read_csv(root / "net_load_independent_q01_q99_2019.csv",
                              parse_dates=["TIMESTAMP"])
    if not actual.TIMESTAMP.equals(copula.TIMESTAMP) or not actual.TIMESTAMP.equals(independent.TIMESTAMP):
        raise ValueError("measured and forecast timestamps do not align")
    week = (actual.TIMESTAMP >= pd.Timestamp("2019-07-01")) & (
        actual.TIMESTAMP < pd.Timestamp("2019-07-08"))
    if week.sum() != 168:
        raise ValueError("expected 168 hourly observations in the selected July week")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH_IN, 4.9),
                                      gridspec_kw={"height_ratios": [2, 1]})
    dates = actual.loc[week, "TIMESTAMP"]
    top.fill_between(dates, copula.loc[week, "q05"], copula.loc[week, "q95"],
                     color="#6699cc", alpha=0.3, label="Copula Q05–Q95")
    top.plot(dates, copula.loc[week, "q50"], color="#174a7e", linewidth=1.3,
             label="Forecast Q50")
    top.plot(dates, actual.loc[week, "net_load_actual_mw"], color="#b23c3c",
             linewidth=1.1, label="Measured net load")
    top.set_ylabel("Net load (MW)")
    top.set_title("Measured German-grid test, 1–7 July 2019 (UTC)", pad=28)
    top.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
               columnspacing=0.9, handlelength=1.6, frameon=False)
    top.grid(alpha=0.2)

    months = actual.TIMESTAMP.dt.month.to_numpy()
    for name, forecast, color in (("Copula", copula, "#174a7e"),
                                  ("Independent", independent, "#b07820")):
        covered = ((actual.net_load_actual_mw >= forecast.q05) &
                   (actual.net_load_actual_mw <= forecast.q95)).to_numpy()
        rates = [100 * covered[months == month].mean() for month in range(1, 13)]
        bottom.plot(range(1, 13), rates, marker="o", markersize=3,
                    linewidth=1.3, label=name, color=color)
    bottom.axhline(90, color="#777777", linestyle="--", linewidth=1,
                   label="Nominal 90%")
    bottom.set(xlabel="2019 month", ylabel="Q05–Q95 coverage (%)",
               xlim=(1, 12), xticks=range(1, 13), ylim=(40, 100))
    bottom.grid(alpha=0.2)
    bottom.legend(loc="lower left", ncol=3, frameon=False)
    fig.tight_layout(pad=1.0, h_pad=1.6)
    destination = root / "measured_net_load_baseline.pdf"
    save_figure(fig, destination)
    plt.close(fig)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(plot(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

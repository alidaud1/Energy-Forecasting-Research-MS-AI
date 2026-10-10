"""Create the v2 ablation figure from frozen metrics; do not rescore."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figure_style import FIGURE_WIDTH_IN, journal_style, save_figure


ROOT = Path(__file__).resolve().parent
# The manuscript includes the figure from the experiment's own directory, so
# write it there.  An earlier revision saved to experiments/figures_v2/ and the
# published copy had to be moved by hand, which meant rerunning this script
# silently left the paper unchanged.
EXPERIMENT = ROOT / "synthetic_holdout"
FIGURES = EXPERIMENT / "figures_v2"
ORDER = ["same_rank", "independent_bounded", "copula_bounded", "copula_unbounded"]
LABELS = ["Same-rank\nsubtraction", "Independent\nsampling", "Copula +\nbounds", "Copula,\nunbounded"]
COLORS = ["#9ca3af", "#60a5fa", "#0f766e", "#e5a64c"]


def label_bars(ax, x, values, fmt):
    """Print each value above its bar, so the figure is readable without the table."""
    span = max(values) - min(min(values), 0)
    for xi, value in zip(x, values):
        ax.annotate(fmt.format(value), (xi, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom")
    return span


def main() -> None:
    result = json.loads((EXPERIMENT / "ablation_results.json").read_text())
    metrics = result["methods"]
    x = np.arange(len(ORDER))

    with journal_style():
        fig, axes = plt.subplots(3, 1, figsize=(FIGURE_WIDTH_IN, 6.2))

        # All three panels are bars over the same four methods.  A line would
        # imply an ordering between what are unordered categories.  Only the
        # loss axis is truncated, and its label says so, because the four
        # scores differ by less than 0.1 MW and a zero-based axis hides that.
        ax = axes[0]
        loss = np.array([metrics[name]["mean_pinball_loss_mw"] for name in ORDER])
        ax.bar(x, loss, color=COLORS)
        ax.set_ylim(2.35, 2.58)
        ax.set_ylabel("MW\n(axis truncated)")
        ax.set_title("Mean pinball loss (lower is better)")
        ax.set_xticks(x, LABELS)
        label_bars(ax, x, loss, "{:.2f}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        # Coverage is stated in per cent on both the axis and the labels; the
        # stored metric is a fraction, so it is scaled once here.
        ax = axes[1]
        coverage = np.array([100.0 * metrics[name]["coverage_q05_q95"] for name in ORDER])
        ax.bar(x, coverage, color=COLORS)
        ax.set_title("Q05\u2013Q95 coverage")
        ax.set_ylabel("Per cent of hours")
        ax.set_xticks(x, LABELS)
        ax.set_ylim(0, 138)
        label_bars(ax, x, coverage, "{:.2f}%")
        ax.axhline(90.0, color="#b91c1c", ls="--", lw=1, label="Nominal 90%")
        ax.legend(loc="upper left", frameon=True, facecolor="white",
                  edgecolor="none", framealpha=1)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        ax = axes[2]
        width = np.array([metrics[name]["mean_width_q05_q95_mw"] for name in ORDER])
        ax.bar(x, width, color=COLORS)
        ax.set_title("Mean Q05\u2013Q95 width (sharpness)")
        ax.set_ylabel("MW")
        ax.set_xticks(x, LABELS)
        ax.set_ylim(0, max(width) * 1.30)
        label_bars(ax, x, width, "{:.2f}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        # The two copula variants agree to the displayed precision.  The caption
        # says why, rather than an in-panel note: every placement tested here
        # collided with a bar or a value label.

        fig.tight_layout(h_pad=1.2)
        save_figure(fig, FIGURES / "fusion_ablation_v2")
        plt.close(fig)


if __name__ == "__main__":
    main()

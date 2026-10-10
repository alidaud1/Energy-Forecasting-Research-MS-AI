"""Draw the architecture figure as vector artwork; export PDF/SVG/PNG/HTML.

Run from the project root:
    MPLCONFIGDIR=/tmp/eaai-mpl agentic/.venv/bin/python -m \
        elsevier_paper.experiments.build_architecture

Landscape composition at the manuscript's text width.  Two forecasting lanes on
the left stay visually separate, because the load and solar marginals are
evaluated separately; everything to the right of the contract is the single
left-to-right joint pipeline.  The figure carries no title: the caption in the
manuscript supplies it.

Ordering in the pipeline is load-bearing and is drawn in the order the code
executes: dependence is estimated from calibration residuals *before* sampling,
because the copula needs the parameter in order to pair the two probability
draws.  Scenarios are then sorted before quantiles are read off them.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np

from .figure_style import FIGURE_WIDTH_IN, journal_style, save_figure

CAPTION = (
    "Architecture linking separate load and solar forecasting workflows to net-load fusion and "
    "decision support. Forecast and residual insets use stored synthetic-benchmark "
    "artifacts; other icons illustrate processing stages."
)

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = Path(__file__).resolve().parent / "synthetic_holdout"
CALIBRATION_END = "2020-01-08"


def sparkline_data():
    """Read the stored synthetic-benchmark artifacts the insets draw from.

    Nothing is recomputed or refitted here: the quantile curves, the recorded
    rank correlation and the net-load forecast are read back exactly as the
    experiment wrote them, so the insets cannot drift from the reported run.
    """
    import pandas as pd

    actual = pd.read_csv(HOLDOUT / "synthetic_actuals.csv", parse_dates=["TIMESTAMP"])
    load = pd.read_csv(HOLDOUT / "load_predictions.csv", parse_dates=["TIMESTAMP"])
    solar = pd.read_csv(HOLDOUT / "solar_predictions.csv", parse_dates=["TIMESTAMP"])
    net = pd.read_csv(HOLDOUT / "joint_run" / "net_load_forecast.csv", parse_dates=["TIMESTAMP"])
    manifest = json.loads((HOLDOUT / "synthetic_benchmark_manifest.json").read_text())

    calib = (actual.TIMESTAMP < CALIBRATION_END).to_numpy()
    window = slice(0, 72)   # first three days of the held-out week

    def band(frame, rows):
        return (frame.q05.to_numpy()[rows], frame.q50.to_numpy()[rows],
                frame.q95.to_numpy()[rows])

    test_rows = np.flatnonzero(~calib)[window]
    capacity = float(manifest["capacity_mw"])
    rho = float(manifest["calibration_rank_correlation"])

    # Paired calibration-period median residuals: the input the copula uses.
    load_resid = actual.load_actual_MW.to_numpy()[calib] - load.q50.to_numpy()[calib]
    solar_resid = (actual.solar_actual_MW.to_numpy()[calib]
                   - solar.q50.to_numpy()[calib] * capacity)

    net_rows = np.arange(72)
    return {
        "load": band(load, test_rows),
        "solar": tuple(v * capacity for v in band(solar, test_rows)),
        "net": band(net, net_rows),
        "resid": (load_resid, solar_resid),
        "rho": rho,
        "reserve": (net.q50.to_numpy()[net_rows], net.q95.to_numpy()[net_rows]),
    }

# Landscape drawing units, with three aligned panels and spaced text baselines.
W, H = 480.0, 336.0

INK = "#22303d"          # body text
MUTED = "#5d6d7b"        # secondary text
NAVY = "#1d3f5e"         # load lane
TEAL = "#17706b"         # solar lane and joint pipeline
RULE = "#aab6c2"         # hairlines
SOFT_NAVY = "#eef3f8"
SOFT_TEAL = "#eaf4f2"
AMBER = "#8a6a2f"
SOFT_AMBER = "#f4f1ea"

LABEL_PT = 7.6
SMALL_PT = 6.3
NOTE_PT = 5.5


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W / 72.27, H / 72.27))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, W), ylim=(H, 0))
        self.ax.axis("off")

    # -- primitives ---------------------------------------------------------
    def box(self, cx, cy, w, h, edge, fill="white", lw=0.7, dashed=False, radius=2.6):
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2,
            linestyle=(0, (2.6, 1.8)) if dashed else "solid")
        self.ax.add_patch(patch)
        return patch

    def label(self, cx, cy, title, detail=None, color=INK, title_pt=LABEL_PT):
        """Bold label with an optional muted second line beneath it."""
        if detail is None:
            self.ax.text(cx, cy, title, fontsize=title_pt, fontweight="bold",
                         color=color, ha="center", va="center", zorder=5,
                         linespacing=1.22)
            return
        lines = title.count("\n") + 1
        offset = 7.0 * lines
        self.ax.text(cx, cy - offset + 3, title, fontsize=title_pt,
                     fontweight="bold", color=color, ha="center", va="center",
                     zorder=5, linespacing=1.22)
        self.ax.text(cx, cy + 10.5, detail, fontsize=SMALL_PT, color=MUTED,
                     ha="center", va="center", zorder=5, linespacing=1.25)

    def arrow(self, start, end, color=INK, dashed=False, lw=0.7, rad=0.0):
        self.ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=4.4,
            linewidth=lw, color=color, zorder=4,
            shrinkA=0, shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=(0, (2.6, 1.8)) if dashed else "solid"))

    def elbow(self, points, color=INK, dashed=False, lw=0.7):
        """Orthogonal polyline with a single arrowhead on the final leg."""
        points = [tuple(map(float, p)) for p in points]
        style = (0, (2.6, 1.8)) if dashed else "solid"
        for a, b in zip(points[:-2], points[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                         linestyle=style, zorder=3, solid_capstyle="round")
        self.arrow(points[-2], points[-1], color=color, dashed=dashed, lw=lw)


# --- line icons ------------------------------------------------------------
# Drawn rather than imported so the artwork stays vector and self-contained.

def icon_pylon(ax, x, y, s, c):
    """Transmission tower: load data, the grid-side input."""
    ax.plot([x, x], [y - s * .5, y + s * .5], color=c, lw=0.6, zorder=5)
    for k, (dy, half) in enumerate(((-.32, .40), (-.06, .30), (.2, .20))):
        yy = y + s * dy
        ax.plot([x - s * half, x + s * half], [yy, yy], color=c, lw=0.55, zorder=5)
        for sx in (-1, 1):
            ax.plot([x, x + sx * s * half], [yy - s * .1, yy], color=c, lw=0.45, zorder=5)
    ax.plot([x - s * .22, x], [y + s * .5, y + s * .28], color=c, lw=0.55, zorder=5)
    ax.plot([x + s * .22, x], [y + s * .5, y + s * .28], color=c, lw=0.55, zorder=5)


def icon_meter(ax, x, y, s, c):
    ax.add_patch(FancyBboxPatch((x - s * .5, y - s * .55), s, s * 1.1,
                                boxstyle="round,pad=0,rounding_size=2.2",
                                lw=0.6, edgecolor=c, facecolor="none", zorder=5))
    ax.plot([x - s * .26, x + s * .26], [y - s * .2, y - s * .2], color=c, lw=0.6, zorder=5)
    ax.add_patch(Polygon([[x + s * .06, y - s * .05], [x - s * .14, y + s * .34],
                          [x + s * .01, y + s * .34], [x - s * .05, y + s * .5]],
                         closed=False, fill=False, lw=0.6, edgecolor=c, zorder=5))


def icon_sun(ax, x, y, s, c):
    ax.add_patch(Circle((x, y), s * .3, lw=0.6, edgecolor=c, facecolor="none", zorder=5))
    for k in range(8):
        a = k * np.pi / 4
        ax.plot([x + np.cos(a) * s * .44, x + np.cos(a) * s * .62],
                [y + np.sin(a) * s * .44, y + np.sin(a) * s * .62],
                color=c, lw=0.6, zorder=5, solid_capstyle="round")


def icon_graph(ax, x, y, s, c):
    """Small node-and-edge glyph for a workflow."""
    top, mid = (x, y - s * .42), (x, y - s * .05)
    legs = [(x - s * .42, y + s * .42), (x, y + s * .42), (x + s * .42, y + s * .42)]
    for p in (top, mid, *legs):
        ax.add_patch(Circle(p, s * .1, lw=0.55, edgecolor=c, facecolor="white", zorder=6))
    ax.plot([top[0], mid[0]], [top[1] + s * .1, mid[1] - s * .1], color=c, lw=0.55, zorder=5)
    for p in legs:
        ax.plot([mid[0], p[0]], [mid[1] + s * .1, p[1] - s * .1], color=c, lw=0.55, zorder=5)


def icon_fan(ax, x, y, s, c):
    """Median line inside a widening band: a quantile forecast."""
    t = np.linspace(0, 1, 40)
    med = y + s * .28 - t * s * .5
    spread = s * (.06 + .34 * t)
    xs = x - s * .55 + t * s * 1.1
    ax.fill_between(xs, med - spread, med + spread, color=c, alpha=.16, lw=0, zorder=4)
    ax.plot(xs, med, color=c, lw=0.6, zorder=5)


def icon_contract(ax, x, y, s, c):
    """Document with a check and a shield: the typed, validated boundary."""
    ax.add_patch(FancyBboxPatch((x - s * .4, y - s * .5), s * .62, s,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                lw=0.6, edgecolor=c, facecolor="white", zorder=5))
    for dy in (-.22, -.02, .18):
        ax.plot([x - s * .28, x + s * .06], [y + s * dy, y + s * dy],
                color=c, lw=0.45, zorder=5)
    sx, sy = x + s * .28, y + s * .12
    ax.add_patch(Polygon([[sx - s * .18, sy - s * .2], [sx + s * .18, sy - s * .2],
                          [sx + s * .18, sy + s * .06], [sx, sy + s * .26],
                          [sx - s * .18, sy + s * .06]],
                         closed=True, lw=0.55, edgecolor=c, facecolor="white", zorder=6))
    ax.plot([sx - s * .08, sx - s * .01, sx + s * .11],
            [sy + s * .0, sy - s * .08, sy + s * .12], color=c, lw=0.55, zorder=7)


def icon_clock(ax, x, y, s, c):
    ax.add_patch(Circle((x, y), s * .46, lw=0.6, edgecolor=c, facecolor="none", zorder=5))
    ax.plot([x, x], [y, y - s * .28], color=c, lw=0.6, zorder=5, solid_capstyle="round")
    ax.plot([x, x + s * .22], [y, y], color=c, lw=0.6, zorder=5, solid_capstyle="round")


def icon_scatter(ax, x, y, s, c):
    """Tilted cloud inside an ellipse: estimated rank dependence."""
    ax.add_patch(Arc((x, y), s * 1.05, s * .54, angle=-28, theta1=0, theta2=360,
                     lw=0.6, edgecolor=c, zorder=5))
    rng = np.random.default_rng(7)
    u = rng.normal(0, s * .22, 9)
    v = u * .55 + rng.normal(0, s * .07, 9)
    ang = np.deg2rad(-28)
    ax.scatter(x + u * np.cos(ang) - v * np.sin(ang),
               y + u * np.sin(ang) + v * np.cos(ang),
               s=0.9, color=c, zorder=6, linewidths=0)


def icon_dice(ax, x, y, s, c):
    ax.add_patch(FancyBboxPatch((x - s * .5, y - s * .44), s * .82, s * .82,
                                boxstyle="round,pad=0,rounding_size=2.2",
                                lw=0.6, edgecolor=c, facecolor="white", zorder=5))
    for dx, dy in ((-.26, -.2), (0, 0), (.26, .2)):
        ax.add_patch(Circle((x - s * .09 + dx * s, y - s * .03 + dy * s), s * .055,
                            color=c, zorder=6))


def icon_balance(ax, x, y, s, c):
    ax.plot([x, x], [y - s * .4, y + s * .38], color=c, lw=0.6, zorder=5)
    ax.plot([x - s * .45, x + s * .45], [y - s * .26, y - s * .26], color=c, lw=0.6, zorder=5)
    for sx in (-1, 1):
        ax.plot([x + sx * s * .45, x + sx * s * .3], [y - s * .26, y + s * .04],
                color=c, lw=0.55, zorder=5)
        ax.plot([x + sx * s * .6, x + sx * s * .3], [y + s * .04, y + s * .04],
                color=c, lw=0.55, zorder=5)
    ax.plot([x - s * .3, x + s * .3], [y + s * .38, y + s * .38], color=c, lw=0.6, zorder=5)


def icon_sort(ax, x, y, s, c):
    for k, wdt in enumerate((.22, .36, .5, .64)):
        yy = y - s * .33 + k * s * .22
        ax.plot([x - s * .5, x - s * .5 + s * wdt], [yy, yy], color=c, lw=0.7,
                zorder=5, solid_capstyle="round")


def icon_rules(ax, x, y, s, c):
    ax.add_patch(FancyBboxPatch((x - s * .42, y - s * .5), s * .84, s,
                                boxstyle="round,pad=0,rounding_size=2.2",
                                lw=0.6, edgecolor=c, facecolor="none", zorder=5))
    for k in range(3):
        yy = y - s * .27 + k * s * .27
        ax.plot([x - s * .26, x - s * .16], [yy, yy + s * .07], color=c, lw=0.6, zorder=5)
        ax.plot([x - s * .16, x + s * .02], [yy + s * .07, yy - s * .09], color=c, lw=0.6, zorder=5)
        ax.plot([x + s * .1, x + s * .27], [yy, yy], color=c, lw=0.6, zorder=5)


def icon_report(ax, x, y, s, c):
    ax.add_patch(FancyBboxPatch((x - s * .36, y - s * .48), s * .72, s * .96,
                                boxstyle="round,pad=0,rounding_size=2.2",
                                lw=0.6, edgecolor=c, facecolor="none", zorder=5))
    for k, hgt in enumerate((.2, .34, .26)):
        xx = x - s * .18 + k * s * .18
        ax.plot([xx, xx], [y + s * .26, y + s * .26 - s * hgt], color=c, lw=0.8,
                zorder=5, solid_capstyle="round")


def icon_chat(ax, x, y, s, c):
    ax.add_patch(FancyBboxPatch((x - s * .46, y - s * .44), s * .92, s * .66,
                                boxstyle="round,pad=0,rounding_size=3",
                                lw=0.6, edgecolor=c, facecolor="none", zorder=5))
    ax.add_patch(Polygon([[x - s * .16, y + s * .22], [x - s * .06, y + s * .44],
                          [x + s * .04, y + s * .22]], closed=True,
                         facecolor="white", edgecolor=c, lw=0.6, zorder=6))
    for dx in (-.2, 0, .2):
        ax.add_patch(Circle((x + dx * s, y - s * .1), s * .045, color=c, zorder=7))


def icon_loop(ax, x, y, s, c):
    """Circular arrow: the bounded validation-driven retraining cycle."""
    r = s * .40
    ax.add_patch(Arc((x, y), 2 * r, 2 * r, angle=0, theta1=62, theta2=360,
                     lw=0.65, edgecolor=c, zorder=5, linestyle=(0, (2.0, 1.3))))
    a1, a2 = np.deg2rad(74), np.deg2rad(58)
    ax.add_patch(FancyArrowPatch(
        (x + r * np.cos(a1), y + r * np.sin(a1)),
        (x + r * np.cos(a2), y + r * np.sin(a2)),
        arrowstyle="-|>", mutation_scale=3.6, linewidth=0.65, color=c, zorder=6,
        shrinkA=0, shrinkB=0))


def icon_calib(ax, x, y, s, c):
    """Paired residual dots about a zero line."""
    ax.plot([x - s * .5, x + s * .5], [y, y], color=c, lw=0.6, zorder=5)
    rng = np.random.default_rng(3)
    xs = np.linspace(x - s * .4, x + s * .4, 7)
    ax.scatter(xs, y + rng.normal(0, s * .17, 7), s=0.9, color=c, zorder=6, linewidths=0)




# --- inset plots -----------------------------------------------------------
# Small real plots rather than decorative glyphs: each one is drawn from the
# stored artifact the step it sits in actually produces.

def inset(c, cx, cy, w, h):
    """Axes in figure coordinates covering a rectangle of the drawing canvas."""
    ax = c.fig.add_axes([(cx - w / 2) / W, 1 - (cy + h / 2) / H, w / W, h / H], zorder=7)
    ax.set_facecolor("none")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([]), ax.set_yticks([])
    return ax


def spark_fan(c, cx, cy, w, h, band, color):
    lo, mid, hi = band
    ax = inset(c, cx, cy, w, h)
    x = np.arange(len(mid))
    ax.fill_between(x, lo, hi, color=color, alpha=0.20, lw=0)
    ax.plot(x, mid, color=color, lw=0.55)
    ax.set_xlim(0, len(mid) - 1)
    pad = 0.08 * (hi.max() - lo.min() + 1e-9)
    ax.set_ylim(lo.min() - pad, hi.max() + pad)


def spark_scatter(c, cx, cy, w, h, xs, ys, color):
    ax = inset(c, cx, cy, w, h)
    ax.scatter(xs, ys, s=0.6, color=color, alpha=0.55, linewidths=0)
    ax.axhline(0, color=RULE, lw=0.35)
    ax.axvline(0, color=RULE, lw=0.35)


def spark_copula(c, cx, cy, w, h, rho, color):
    """Paired uniforms under the recorded dependence: what the copula produces."""
    ax = inset(c, cx, cy, w, h)
    rng = np.random.default_rng(2026)
    z1, z2 = rng.normal(size=400), rng.normal(size=400)
    g = 2.0 * np.sin(np.pi * rho / 6.0)
    from scipy.stats import norm
    u = norm.cdf(z1)
    v = norm.cdf(g * z1 + np.sqrt(1 - g ** 2) * z2)
    ax.scatter(u, v, s=0.5, color=color, alpha=0.5, linewidths=0)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)


def spark_hist(c, cx, cy, w, h, band, color):
    """Spread of net load across the window: the ensemble the quantiles come from."""
    lo, mid, hi = band
    ax = inset(c, cx, cy, w, h)
    ax.hist(mid, bins=16, color=color, alpha=0.55, lw=0)
    ax.axvline(float(np.median(mid)), color=color, lw=0.6)


def spark_reserve(c, cx, cy, w, h, reserve, color):
    """Q50 path with the Q95 margin above it: the reserve proxy the rules report."""
    q50, q95 = reserve
    ax = inset(c, cx, cy, w, h)
    x = np.arange(len(q50))
    ax.fill_between(x, q50, q95, color=color, alpha=0.22, lw=0)
    ax.plot(x, q50, color=color, lw=0.55)
    peak = int(np.argmax(q50))
    ax.scatter([peak], [q50[peak]], s=2.4, color=color, zorder=5, linewidths=0)
    ax.set_xlim(0, len(q50) - 1)


# --- composition -----------------------------------------------------------
# Three softly shaded panels, with consistent icon and text baselines.
PANEL_X0, PANEL_X1 = 7.0, 473.0


def panel(c, y0, y1, tint, edge, number, heading):
    c.box(240, (y0 + y1) / 2, 466, y1 - y0, edge, tint,
          lw=0.45, radius=5)
    c.ax.add_patch(Circle((20, y0 + 12), 6, facecolor=edge,
                          edgecolor="none", zorder=3))
    c.ax.text(20, y0 + 12, str(number), fontsize=7, color="white",
              fontweight="bold", ha="center", va="center", zorder=4)
    c.ax.text(32, y0 + 12, heading, fontsize=9, fontweight="bold",
              color=edge, ha="left", va="center", zorder=4)


def card(c, x, y, w, h, color, title, detail=None, dashed=False):
    c.box(x, y, w, h, color, "white", lw=0.65, dashed=dashed, radius=4)
    c.ax.text(x, y - h / 2 + 12, title, fontsize=LABEL_PT,
              fontweight="bold", color=INK, ha="center", va="center",
              zorder=6, linespacing=1.2)
    if detail:
        c.ax.text(x, y + h / 2 - 10, detail, fontsize=SMALL_PT,
                  color=MUTED, ha="center", va="center", zorder=6,
                  linespacing=1.2)


def build():
    data = sparkline_data()
    with journal_style():
        c = Canvas()
        panel(c, 5, 127, SOFT_NAVY, NAVY, 1, "Separate load and solar forecasting")
        xs, ys, cw, ch = (53, 150, 247), (52, 100), 78, 42
        for y, color, name, icon, band in (
                (ys[0], NAVY, "Load", icon_pylon, data["load"]),
                (ys[1], TEAL, "Solar", icon_sun, data["solar"])):
            card(c, xs[0], y, cw, ch, color, f"{name} data")
            icon(c.ax, xs[0], y + 8, 13, color)
            card(c, xs[1], y, cw, ch, color, f"{name} forecasting")
            icon_graph(c.ax, xs[1] - 13, y + 8, 13, color)
            icon_loop(c.ax, xs[1] + 16, y + 8, 17, color)
            card(c, xs[2], y, cw, ch, color, f"{name} Q01–Q99")
            spark_fan(c, xs[2], y + 8, 57, 17, band, color)
            for a, b in zip(xs[:-1], xs[1:]):
                c.arrow((a + cw/2, y), (b - cw/2, y), color=color)
        card(c, 396, 76, 128, 82, MUTED, "Calibration residuals",
             "calibration data;\nnot held-out test data")
        spark_scatter(c, 396, 75, 74, 26, *data["resid"], MUTED)

        panel(c, 142, 237, SOFT_TEAL, TEAL, 2, "Fusion agent · joint net-load uncertainty")
        fx, fy, fw, fh = (52, 127, 202, 277, 352, 427), 196, 64, 64
        titles = ("Forecast\ncontract", "Time & unit\nalignment",
                  "Estimate\ndependence", "Monte Carlo\nsampling",
                  "Net-load\nscenarios", "Net-load\nQ01–Q99")
        details = ("Validate inputs", "Common hours; MW", "Residual ranks",
                   "Gaussian copula", "N = L − S;\nnegatives kept", "Sort scenarios")
        for x, title, detail in zip(fx, titles, details):
            card(c, x, fy, fw, fh, TEAL, title, detail)
        icon_contract(c.ax, fx[0], fy + 1, 16, TEAL)
        icon_clock(c.ax, fx[1], fy + 1, 14, TEAL)
        spark_scatter(c, fx[2], fy + 1, 36, 17, *data["resid"], TEAL)
        icon_dice(c.ax, fx[3], fy + 1, 15, TEAL)
        icon_balance(c.ax, fx[4], fy + 1, 17, TEAL)
        spark_fan(c, fx[5], fy + 1, 40, 17, data["net"], TEAL)
        for a, b in zip(fx[:-1], fx[1:]):
            c.arrow((a + fw/2, fy), (b - fw/2, fy), color=TEAL)

        # Forecast streams join on an outside bus and enter the contract at its left edge.
        c.elbow([(286, 52), (310, 52), (310, 130), (4, 130)], color=NAVY)
        c.elbow([(286, 100), (300, 100), (300, 137), (4, 137)], color=TEAL)
        c.ax.add_patch(Circle((4, 137), 1.5, color=TEAL, zorder=5))
        c.arrow((4, 130), (4, 196), color=TEAL)
        c.arrow((4, 196), (20, 196), color=TEAL)
        # Calibration is a separate estimation input, not a forecast stream.
        c.elbow([(396, 117), (396, 139), (467, 139),
                 (467, 160), (202, 160), (202, 164)],
                color=MUTED, dashed=True)

        gold = AMBER
        panel(c, 249, 330, "#faf4e8", gold, 3, "Decision agent · support for human review")
        dx, dy = (84, 240, 396), 294
        for x, title, detail, color, dashed in (
            (dx[0], "Deterministic decision rules",
             "Reserve proxy, peaks, ramps,\nwarning flags", gold, False),
            (dx[1], "Decision report", "Indicators & run manifest", gold, False),
            (dx[2], "LLM summary",
             "Summarizes the report only; does\nnot calculate or change results",
             gold, True)):
            card(c, x, dy, 128, 47, color, title, detail, dashed=dashed)
        icon_rules(c.ax, dx[0], dy + 1, 11, gold)
        icon_report(c.ax, dx[1], dy + 1, 12, gold)
        icon_chat(c.ax, dx[2], dy + 1, 12, gold)
        c.arrow((148, dy), (176, dy), color=gold)
        c.arrow((304, dy), (332, dy), color=gold, dashed=True)
        c.elbow([(427, 228), (427, 242), (4, 242),
                 (4, 294), (20, 294)], color=TEAL)

        target = ROOT / "figures" / "01_system_architecture"
        save_figure(c.fig, target)
        plt.close(c.fig)
    write_preview(target)
    print(target.with_suffix(".pdf"))


def write_preview(target: Path) -> None:
    """Wrap the exported SVG in the standalone HTML viewer."""
    svg = target.with_suffix(".svg").read_text(encoding="utf-8")
    svg = svg[svg.index("<svg"):]
    svg = svg.replace(
        "<svg ", '<svg role="img" aria-label="Load and solar forecasting architecture" ', 1)
    template = (ROOT / "architecture_viewer.html.in").read_text(encoding="utf-8")
    page = (template.replace("{{SVG}}", svg)
                    .replace("{{CAPTION}}", html.escape(CAPTION))
                    .replace("{{WIDTH_IN}}", f"{FIGURE_WIDTH_IN:.5f}")
                    .replace("{{WIDTH_DISPLAY}}", f"{FIGURE_WIDTH_IN:.2f}"))
    (ROOT / "architecture_diagram.html").write_text(page, encoding="utf-8")

    audit = {
        "font": "Liberation Sans (Arial-compatible)",
        "label_pt": LABEL_PT, "detail_pt": SMALL_PT,
        "figure_width_inches": W / 72.27, "figure_height_inches": H / 72.27,
        "units": "typographic points at native artwork size; scaled uniformly at manuscript width",
        "insets": "forecast and residual plots use stored benchmark artifacts; processing icons are schematic",
        "feedback_arrows": "bounded retraining self-loops on the two forecasting workflows only",
        "caption": CAPTION,
    }
    (ROOT / "figures" / "architecture_style.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(ROOT / "architecture_diagram.html")


if __name__ == "__main__":
    build()

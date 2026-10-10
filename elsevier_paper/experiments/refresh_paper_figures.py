"""Refresh figure typography from saved results; do not refit or rescore.

Reads frozen predictions and ablation metrics, hashes inputs before/after,
and rewrites the three artwork files referenced by manuscript.tex.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .plot_opsd_baseline import plot
from .run_ablations import _figures

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    synthetic = ROOT / "synthetic_holdout"
    measured = ROOT / "opsd_de_baseline"
    paths = [synthetic / "ablation_results.json", synthetic / "synthetic_actuals.csv",
             synthetic / "joint_run" / "net_load_forecast.csv",
             measured / "actuals_2019.csv", measured / "net_load_copula_q01_q99_2019.csv",
             measured / "net_load_independent_q01_q99_2019.csv"]
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    report = json.loads(paths[0].read_text())
    actual = pd.read_csv(paths[1])
    forecast = pd.read_csv(paths[2])
    outputs = [str(plot(measured))]
    # The bar plot uses archived metric values; the fan plot needs only this frame.
    outputs.extend(_figures(synthetic, actual, {"copula_bounded": forecast}, report["methods"]))
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    if before != after:
        raise RuntimeError("A frozen numerical input changed during figure generation")
    destination = ROOT.parent / "figures" / "plot_style_audit.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps({
        "numerical_inputs_unchanged": True, "sha256": before,
        "figures": outputs, "action": "redraw stored results with uniform artwork typography",
    }, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

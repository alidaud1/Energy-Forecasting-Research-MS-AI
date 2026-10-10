"""Independently recompute measured OPSD scores from the saved CSV artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = np.arange(1, 100, dtype=float) / 100.0
QUANTILES = [f"q{i:02d}" for i in range(1, 100)]


def verify(root: Path) -> dict:
    root = root.resolve()
    report = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    actuals = pd.read_csv(root / "actuals_2019.csv")
    if len(actuals) != 8760 or not np.allclose(
            actuals["load_actual_mw"] - actuals["solar_actual_mw"],
            actuals["net_load_actual_mw"]):
        raise AssertionError("missing or inconsistent measured net-load target")
    stamps = pd.to_datetime(actuals["TIMESTAMP"], errors="raise")
    if stamps.duplicated().any() or not stamps.is_monotonic_increasing:
        raise AssertionError("test timestamps are duplicated or unordered")
    checked = {}
    for label, name, target, key in (
        ("load", "load_q01_q99_2019.csv", "load_actual_mw", "load"),
        ("solar", "solar_q01_q99_2019.csv", "solar_actual_mw", "solar"),
        ("copula", "net_load_copula_q01_q99_2019.csv", "net_load_actual_mw", "net_load"),
        ("independent", "net_load_independent_q01_q99_2019.csv", "net_load_actual_mw", "net_load"),
    ):
        frame = pd.read_csv(root / name)
        if not pd.to_datetime(frame["TIMESTAMP"]).equals(stamps):
            raise AssertionError(f"{label}: forecast and target timestamps differ")
        q = frame[QUANTILES].to_numpy(float)
        if not np.isfinite(q).all() or (np.diff(q, axis=1) < -1e-9).any():
            raise AssertionError(f"{label}: invalid or crossing quantile forecasts")
        y = actuals[target].to_numpy(float)
        error = y[:, None] - q
        pinball = float(np.maximum(LEVELS * error, (LEVELS - 1) * error).mean())
        coverage = float(np.mean((y >= q[:, 4]) & (y <= q[:, 94])))
        width = float(np.mean(q[:, 94] - q[:, 4]))
        stored = report[key] if key != "net_load" else report[key][label]
        for name_metric, value in (("mean_pinball_loss_mw", pinball),
                                   ("coverage_q05_q95", coverage),
                                   ("mean_q05_q95_width_mw", width)):
            if not np.isclose(value, stored[name_metric], rtol=1e-10, atol=1e-10):
                raise AssertionError(f"{label}: {name_metric} disagrees with report")
        checked[label] = {"pinball_mw": round(pinball, 2),
                          "coverage_percent": round(100 * coverage, 2),
                          "width_mw": round(width, 2)}
    for method in ("copula", "independent"):
        decisions = json.loads((root / f"decisions_{method}_2019.json").read_text())
        if len(decisions) != 12:
            raise AssertionError(f"{method}: expected 12 monthly decision summaries")
    return {"verified_hours": len(actuals), "scores": checked,
            "monthly_decision_reports_per_method": 12}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

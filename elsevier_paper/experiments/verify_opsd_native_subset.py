"""Recalculate native-subset 2019 scores from saved, same-timestamp CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_opsd_baseline import _score
from fusion.forecast_contract import QUANTILE_COLUMNS


def verify(root: Path) -> dict:
    root = root.resolve()
    report = json.loads((root / "evaluation.json").read_text())
    actuals = pd.read_csv(root / "actuals_2019.csv")
    if len(actuals) != 8760 or not np.allclose(
            actuals.load_actual_mw - actuals.solar_actual_mw,
            actuals.net_load_actual_mw):
        raise AssertionError("incomplete or inconsistent measured actuals")
    stamps = pd.to_datetime(actuals.TIMESTAMP)
    if stamps.duplicated().any() or not stamps.is_monotonic_increasing:
        raise AssertionError("invalid test timestamp order")
    checked = {}
    items = (
        ("load", "load_GBR_q01_q99_2019.csv", "load_actual_mw"),
        ("solar", "solar_XGBoost_q01_q99_2019.csv", "solar_actual_mw"),
        ("copula", "net_load_copula_q01_q99_2019.csv", "net_load_actual_mw"),
        ("independent", "net_load_independent_q01_q99_2019.csv", "net_load_actual_mw"),
    )
    for label, name, target in items:
        frame = pd.read_csv(root / name)
        if not pd.to_datetime(frame.TIMESTAMP).equals(stamps):
            raise AssertionError(f"{label} timestamps do not match actuals")
        q = frame[list(QUANTILE_COLUMNS)].to_numpy(float)
        if not np.isfinite(q).all() or (np.diff(q, axis=1) < -1e-9).any():
            raise AssertionError(f"{label} quantiles invalid or crossing")
        score = _score(actuals[target].to_numpy(float), q)
        stored = report[label] if label in ("load", "solar") else report["net_load"][label]
        for metric, measured in score.items():
            if not np.isclose(measured, stored[metric], rtol=1e-10, atol=1e-10):
                raise AssertionError(f"{label} {metric} differs from saved score")
        checked[label] = {"pinball_mw": round(score["mean_pinball_loss_mw"], 2),
                          "coverage_percent": round(100 * score["coverage_q05_q95"], 2),
                          "width_mw": round(score["mean_q05_q95_width_mw"], 2)}
    for label in ("copula", "independent"):
        monthly = json.loads((root / f"decisions_{label}_2019.json").read_text())
        if len(monthly) != 12:
            raise AssertionError(f"{label} has incomplete decision reports")
    for name in ("Load_GBR_OPSD_DE.pkl", "XGBoost_best.pkl"):
        if not (root / "checkpoints" / name).is_file():
            raise AssertionError(f"missing isolated model checkpoint {name}")
    return {"verified_hours": 8760, "scores": checked, "monthly_decisions": 12}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

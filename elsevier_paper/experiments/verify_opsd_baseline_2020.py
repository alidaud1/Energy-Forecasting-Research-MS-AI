"""Independently recompute saved 2020 simple-baseline scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fusion.forecast_contract import QUANTILE_COLUMNS
from .run_opsd_baseline import _score


def verify(root: Path) -> dict:
    root = root.resolve()
    report = json.loads((root / "evaluation.json").read_text())
    actual = pd.read_csv(root / "actuals_2020.csv")
    stamps = pd.to_datetime(actual.TIMESTAMP, utc=True)
    if len(stamps) != 4367 or stamps.duplicated().any() or \
            not stamps.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all() or \
            not np.allclose(actual.load_actual_mw - actual.solar_actual_mw,
                            actual.net_load_actual_mw):
        raise AssertionError("invalid 2020 measured target or timestamps")
    checked = {}
    for label, filename, target, stored in (
        ("load", "load_q01_q99_2020.csv", "load_actual_mw", report["load"]),
        ("solar", "solar_q01_q99_2020.csv", "solar_actual_mw", report["solar"]),
        ("copula", "net_load_copula_q01_q99_2020.csv", "net_load_actual_mw",
         report["net_load"]["copula"]),
        ("independent", "net_load_independent_q01_q99_2020.csv", "net_load_actual_mw",
         report["net_load"]["independent"]),
    ):
        frame = pd.read_csv(root / filename)
        if not pd.to_datetime(frame.TIMESTAMP, utc=True).equals(stamps):
            raise AssertionError(f"{label}: timestamp mismatch")
        quantiles = frame[list(QUANTILE_COLUMNS)].to_numpy(float)
        if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < -1e-9).any():
            raise AssertionError(f"{label}: invalid quantiles")
        result = _score(actual[target].to_numpy(float), quantiles)
        for metric, value in result.items():
            if not np.isclose(value, stored[metric], atol=1e-10, rtol=1e-10):
                raise AssertionError(f"{label}: {metric} mismatch")
        checked[label] = round(result["mean_pinball_loss_mw"], 2)
    for method in ("copula", "independent"):
        decisions = json.loads((root / f"decisions_{method}_2020.json").read_text())
        if [item["month"] for item in decisions] != list(range(1, 7)):
            raise AssertionError(f"{method}: monthly decisions incomplete")
    return {"verified_hours": len(stamps), "pinball_mw": checked,
            "monthly_decision_reports": 12}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

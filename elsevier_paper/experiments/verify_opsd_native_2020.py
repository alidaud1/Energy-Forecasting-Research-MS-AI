"""Independently recompute every saved 2020 native-port score."""

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
    n = report["test_hours"]
    if n != 4367 or len(actual) != n or not np.allclose(
            actual.load_actual_mw - actual.solar_actual_mw,
            actual.net_load_actual_mw):
        raise AssertionError("invalid 2020 target records")
    stamp = pd.to_datetime(actual.TIMESTAMP, utc=True)
    if stamp.duplicated().any() or not stamp.is_monotonic_increasing or \
            not stamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise AssertionError("2020 timestamps incomplete or misordered")
    out = {}
    for label in ("raw", "calibrated"):
        scores = {}
        for name, target, stored in (
            (f"load_{label}_q01_q99_2020.csv", "load_actual_mw",
             report["variants"][label]["load"]),
            (f"solar_{label}_q01_q99_2020.csv", "solar_actual_mw",
             report["variants"][label]["solar"]),
            *((f"net_load_{label}_{method}_q01_q99_2020.csv",
               "net_load_actual_mw", report["variants"][label]["net_load"][method])
              for method in ("copula", "independent"))):
            frame = pd.read_csv(root / name)
            if not pd.to_datetime(frame.TIMESTAMP, utc=True).equals(stamp):
                raise AssertionError(f"{name}: timestamps differ")
            q = frame[list(QUANTILE_COLUMNS)].to_numpy(float)
            if not np.isfinite(q).all() or (np.diff(q, axis=1) < -1e-9).any():
                raise AssertionError(f"{name}: nonfinite or crossing quantiles")
            result = _score(actual[target].to_numpy(float), q)
            for metric, value in result.items():
                if not np.isclose(value, stored[metric], rtol=1e-10, atol=1e-10):
                    raise AssertionError(f"{name}: saved {metric} does not match")
            scores[name] = round(result["mean_pinball_loss_mw"], 2)
        for method in ("copula", "independent"):
            decisions = json.loads((root / f"decisions_{label}_{method}_2020.json").read_text())
            if [item["month"] for item in decisions] != list(range(1, 7)):
                raise AssertionError(f"{label}/{method}: missing monthly decisions")
        out[label] = scores
    return {"verified_hours": n, "scores_pinball_mw": out,
            "decision_reports": 12}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

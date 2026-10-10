"""Paired 2020 net-load pinball comparisons with HAC uncertainty.

The individual hour is the paired unit.  The Newey--West long-run variance
uses Bartlett weights; a 168-hour lag is the declared primary setting and
24 hours is a sensitivity setting.  This exploratory calculation does not
turn the post-hoc calibration choice into a preregistered test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from fusion.forecast_contract import QUANTILE_COLUMNS
from .run_opsd_baseline import LEVELS


def hourly_pinball(actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.shape != (len(actual), 99) or not np.isfinite(actual).all() or \
            not np.isfinite(quantiles).all():
        raise ValueError("actual and quantile arrays do not align")
    e = actual[:, None] - quantiles
    return np.maximum(LEVELS * e, (LEVELS - 1) * e).mean(axis=1)


def hac_test(difference: np.ndarray, lag: int) -> dict:
    """Two-sided HAC test of zero mean paired loss difference, A minus B."""
    d = np.asarray(difference, dtype=float)
    n = len(d)
    if n < 2 * lag or lag < 0 or not np.isfinite(d).all():
        raise ValueError("invalid loss differential or HAC lag")
    centered = d - d.mean()
    long_run = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        long_run += 2 * (1 - k / (lag + 1)) * gamma
    if long_run < -1e-8:
        raise ValueError("negative HAC long-run variance")
    se = float(np.sqrt(max(long_run, 0.0) / n))
    mean = float(d.mean())
    z = mean / se if se > 0 else np.inf if mean != 0 else 0.0
    return {"n_hours": n, "mean_loss_A_minus_B_mw": mean,
            "hac_lag_hours": lag, "standard_error_mw": se,
            "ci95_mw": [mean - 1.96 * se, mean + 1.96 * se],
            "two_sided_p_normal": float(2 * norm.sf(abs(z)))}


def compare(baseline: Path, native: Path, output: Path) -> dict:
    baseline, native, output = (p.resolve() for p in (baseline, native, output))
    base_actual = pd.read_csv(baseline / "actuals_2020.csv")
    native_actual = pd.read_csv(native / "actuals_2020.csv")
    if not base_actual.TIMESTAMP.equals(native_actual.TIMESTAMP) or \
            not np.allclose(base_actual.net_load_actual_mw,
                            native_actual.net_load_actual_mw):
        raise ValueError("baselines and native port use different 2020 target hours")
    stamps = pd.to_datetime(base_actual.TIMESTAMP, utc=True)
    if len(stamps) != 4367 or stamps.duplicated().any():
        raise ValueError("incomplete 2020 test")
    y = base_actual.net_load_actual_mw.to_numpy(float)
    scores = {}
    for label, root, name in (
        ("baseline_copula", baseline, "net_load_copula_q01_q99_2020.csv"),
        ("native_raw_copula", native, "net_load_raw_copula_q01_q99_2020.csv"),
        ("native_calibrated_copula", native, "net_load_calibrated_copula_q01_q99_2020.csv"),
        ("baseline_independent", baseline, "net_load_independent_q01_q99_2020.csv"),
        ("native_raw_independent", native, "net_load_raw_independent_q01_q99_2020.csv"),
        ("native_calibrated_independent", native, "net_load_calibrated_independent_q01_q99_2020.csv"),
    ):
        frame = pd.read_csv(root / name)
        if not pd.to_datetime(frame.TIMESTAMP, utc=True).equals(stamps):
            raise ValueError(f"{label} timestamps differ")
        q = frame[list(QUANTILE_COLUMNS)].to_numpy(float)
        if (np.diff(q, axis=1) < -1e-9).any():
            raise ValueError(f"{label} quantiles cross")
        scores[label] = hourly_pinball(y, q)
    pairs = (("native_raw_copula", "native_calibrated_copula"),
             ("baseline_copula", "native_raw_copula"),
             ("baseline_copula", "native_calibrated_copula"),
             ("native_raw_independent", "native_calibrated_independent"),
             ("baseline_independent", "native_raw_independent"))
    results = {}
    for a, b in pairs:
        diff = scores[a] - scores[b]
        results[f"{a}_minus_{b}"] = {"hac_168h": hac_test(diff, 168),
                                       "hac_24h_sensitivity": hac_test(diff, 24)}
    report = {"test_hours": len(y),
              "interpretation": "Exploratory paired Q01–Q99 pinball comparisons; negative A-minus-B favors A. HAC normal approximation with Bartlett weights. No multiple-testing correction or operational decision-cost claim.",
              "mean_pinball_mw": {k: float(v.mean()) for k, v in scores.items()},
              "comparisons": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.native, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Checkpoint replay verification and paired inference for fresh load models."""
import json

import numpy as np
import pandas as pd

from .run_full_budget_load import DEFAULT, QCOLS, LEVELS, verify_model
from .audit_solar_statistics import hac_mean


def run():
    losses, reports = {}, {}
    first = None
    for model in ("Linear", "GBR"):
        reports[model] = verify_model(DEFAULT / model, DEFAULT)
        frame = pd.read_csv(DEFAULT / model / "test_q01_q99.csv")
        keys = frame[["TIMESTAMP", "actual_LOAD"]]
        if first is None:
            first = keys
        else:
            pd.testing.assert_frame_equal(first, keys)
        error = frame.actual_LOAD.to_numpy()[:, None]-frame[QCOLS].to_numpy()
        losses[model] = np.maximum(LEVELS*error, (LEVELS-1)*error).mean(axis=1)
        # 273 days = 39 complete nonoverlapping weeks. Whole-week sampling
        # retains within-week dependence, not dependence between weeks.
        weeks = losses[model].reshape(39, 168).mean(axis=1)
        draws = np.random.default_rng(2026).choice(weeks, (5000, 39)).mean(axis=1)
        reports[model]["test_pinball_shared_week_ci95"] = np.quantile(draws, [.025, .975]).tolist()
    difference = losses["Linear"]-losses["GBR"]
    report = {"experiment": "fresh_full_budget_strict24h_load", "models": reports,
              "comparison": "Linear minus GBR hourly mean Q01-Q99 pinball; positive favors GBR",
              "primary_hac_lag168": hac_mean(difference, 168),
              "sensitivity_hac_lag24": hac_mean(difference, 24),
              "bootstrap": {"draws": 5000, "seed": 2026, "block_hours": 168,
                            "limit": "preserves within-week dependence, not between-week dependence"}}
    (DEFAULT / "verified_statistical_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()

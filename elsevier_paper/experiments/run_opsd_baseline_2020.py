"""Reproduce the 2019 simple baseline and extend it to the 2020 holdout.

The 2017--June 2018 point models and July--December 2018 residual calibration
are reconstructed with the original settings. Reproduction of the already
saved 2019 forecasts is checked before the model is applied to 2020. No 2019
or 2020 target participates in fitting or calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from fusion.decision_agent import make_decisions
from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS
from fusion.fusion_agent import fuse_forecasts

from .run_opsd_baseline import _features, _forecast_frame, _hash, _quantiles_by_hour, _score


def run(history: Path, old_results: Path, holdout: Path, output: Path,
        *, scenarios: int = 1000, seed: int = 2026) -> dict:
    history, old_results, holdout, output = (p.resolve() for p in
                                             (history, old_results, holdout, output))
    if scenarios < 100:
        raise ValueError("at least 100 scenarios required")
    frame = pd.read_csv(history)
    features, local, solar_ratio = _features(frame)
    available = features.notna().all(axis=1)
    train = ((local >= pd.Timestamp("2017-01-01", tz="Europe/Berlin")) &
             (local < pd.Timestamp("2018-07-01", tz="Europe/Berlin")) & available).to_numpy()
    cal = ((local >= pd.Timestamp("2018-07-01", tz="Europe/Berlin")) &
           (local < pd.Timestamp("2019-01-01", tz="Europe/Berlin")) & available).to_numpy()
    old = ((local >= pd.Timestamp("2019-01-01", tz="Europe/Berlin")) &
           (local < pd.Timestamp("2020-01-01", tz="Europe/Berlin")) & available).to_numpy()
    if (train.sum(), cal.sum(), old.sum()) != (12934, 4417, 8760):
        raise ValueError("historical baseline split changed")
    load_cols = [c for c in features if not c.startswith("solar_") and c != "capacity_lag25"]
    solar_cols = [c for c in features if not c.startswith("load_")]
    kwargs = dict(max_iter=150, max_leaf_nodes=31, learning_rate=0.06,
                  min_samples_leaf=30, l2_regularization=1.0, random_state=seed)
    load_model = HistGradientBoostingRegressor(**kwargs).fit(
        features.loc[train, load_cols], frame.loc[train, "load_mw"])
    solar_model = HistGradientBoostingRegressor(**kwargs).fit(
        features.loc[train, solar_cols], solar_ratio.loc[train])
    cal_l_point = load_model.predict(features.loc[cal, load_cols])
    cal_s_point = solar_model.predict(features.loc[cal, solar_cols])
    cal_hour = local.loc[cal].dt.hour.to_numpy()
    cal_l_actual = frame.loc[cal, "load_mw"].to_numpy()
    cal_cap = features.loc[cal, "capacity_lag25"].to_numpy()
    cal_s_actual = frame.loc[cal, "solar_mw"].to_numpy()

    def forecast(l_feat: pd.DataFrame, s_feat: pd.DataFrame,
                 hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        l_point = load_model.predict(l_feat[load_cols])
        s_point = solar_model.predict(s_feat[solar_cols])
        l_q = _quantiles_by_hour(l_point, cal_l_point, cal_l_actual,
                                 hours, cal_hour)
        s_q_ratio = _quantiles_by_hour(
            s_point, cal_s_point, cal_s_actual / cal_cap,
            hours, cal_hour, high=1.0)
        return l_q, s_q_ratio * s_feat.capacity_lag25.to_numpy()[:, None]

    # Guard against silent drift in fitting or feature semantics.
    old_l, old_s = forecast(features.loc[old, load_cols],
                            features.loc[old, solar_cols], local.loc[old].dt.hour.to_numpy())
    qnames = list(QUANTILE_COLUMNS)
    old_l_saved = pd.read_csv(old_results / "load_q01_q99_2019.csv")[qnames].to_numpy()
    old_s_saved = pd.read_csv(old_results / "solar_q01_q99_2019.csv")[qnames].to_numpy()
    max_reproduction_difference = float(max(np.max(np.abs(old_l - old_l_saved)),
                                        np.max(np.abs(old_s - old_s_saved))))
    if max_reproduction_difference > 1e-7:
        raise ValueError(f"2019 baseline was not reproduced: {max_reproduction_difference} MW")
    cal_l_q = _quantiles_by_hour(cal_l_point, cal_l_point, cal_l_actual,
                                 cal_hour, cal_hour)
    cal_s_q = _quantiles_by_hour(cal_s_point, cal_s_point, cal_s_actual / cal_cap,
                                 cal_hour, cal_hour, high=1.0) * cal_cap[:, None]
    rho = float(spearmanr(cal_l_actual - cal_l_q[:, 49],
                          cal_s_actual - cal_s_q[:, 49]).statistic)
    old_report = json.loads((old_results / "evaluation.json").read_text())
    if not np.isclose(rho, old_report["calibration_rank_correlation"], atol=1e-12):
        raise ValueError("2019 baseline dependence coefficient not reproduced")

    l_test = pd.read_csv(holdout / "load_test_2020.csv")
    s_test = pd.read_csv(holdout / "solar_test_2020.csv")
    actual = pd.read_csv(holdout / "actuals_2020.csv")
    if not l_test.TIMESTAMP.equals(s_test.TIMESTAMP) or \
            not l_test.TIMESTAMP.equals(actual.TIMESTAMP) or len(actual) != 4367:
        raise ValueError("2020 timestamps or row count differ")
    stamp = pd.to_datetime(l_test.TIMESTAMP, utc=True)
    hour = stamp.dt.tz_convert("Europe/Berlin").dt.hour.to_numpy()
    l_q, s_q = forecast(l_test, s_test, hour)
    l_frame, s_frame = _forecast_frame(stamp, l_q), _forecast_frame(stamp, s_q)
    bound = float(json.loads((holdout / "manifest.json").read_text())["capacity_proxy_mw"])
    l_bundle = ForecastBundle("OPSD-DE simple load baseline", l_frame, "MW", "DE", "24h",
                              physical_min=0.0)
    s_bundle = ForecastBundle("OPSD-DE simple solar baseline", s_frame, "MW", "DE", "24h",
                              metadata={"capacity_mw": bound},
                              physical_min=0.0, physical_max=bound)
    output.mkdir(parents=True, exist_ok=True)
    l_frame.to_csv(output / "load_q01_q99_2020.csv", index=False)
    s_frame.to_csv(output / "solar_q01_q99_2020.csv", index=False)
    actual.to_csv(output / "actuals_2020.csv", index=False)
    months = stamp.dt.tz_convert("Europe/Berlin").dt.month.to_numpy()
    net_results = {}
    for method, dependence in (("copula", rho), ("independent", 0.0)):
        pieces, decisions = [], []
        for m in range(1, 7):
            take = months == m
            fused = fuse_forecasts(
                l_bundle.replace_frame(l_frame.loc[take].reset_index(drop=True)),
                s_bundle.replace_frame(s_frame.loc[take].reset_index(drop=True)),
                n_scenarios=scenarios, seed=seed + m,
                dependence=dependence, output_location="DE")
            decision = make_decisions(fused)
            pieces.append(fused.forecast)
            decisions.append({"month": m, "summary": decision.summary,
                              "risk_flags": decision.risk_flags,
                              "recommendations": decision.recommendations})
        merged = pd.concat(pieces, ignore_index=True)
        if not pd.to_datetime(merged.TIMESTAMP, utc=True).equals(stamp):
            raise ValueError("2020 monthly fused timestamps differ")
        merged.to_csv(output / f"net_load_{method}_q01_q99_2020.csv", index=False)
        (output / f"decisions_{method}_2020.json").write_text(json.dumps(decisions, indent=2) + "\n")
        net_results[method] = _score(actual.net_load_actual_mw.to_numpy(),
                                     merged[qnames].to_numpy())
    report = {
        "experiment": "OPSD_DE_reproduced_simple_baseline_2020H1",
        "not_existing_GEFCom_forecasting_agents": True,
        "history_sha256": _hash(history),
        "holdout_manifest_sha256": _hash(holdout / "manifest.json"),
        "old_2019_reproduction_max_abs_difference_mw": max_reproduction_difference,
        "train_hours": int(train.sum()), "calibration_hours": int(cal.sum()),
        "test_hours": len(actual), "rank_correlation": rho,
        "scenarios_per_hour": scenarios, "seed": seed,
        "capacity_caveat": "2020 solar capacity unavailable; last published 2019 value used as a fixed input proxy.",
        "load": _score(actual.load_actual_mw.to_numpy(), l_q),
        "solar": _score(actual.solar_actual_mw.to_numpy(), s_q),
        "net_load": net_results,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("old_results", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run(args.history, args.old_results, args.holdout,
                         args.output, scenarios=args.scenarios, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

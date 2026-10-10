"""Measured German-grid baseline: day-ahead marginals, fusion and decisions.

This is a deliberately simple, reproducible measured-data *baseline*, not a
claim that the existing GEFCom-trained forecasting agents were retrained.
Models use only calendar features and observations from at least 25 hours
before each target interval start, so hourly intervals have ended by a
strict 24-hour forecast issue time. Residual quantiles and load--solar dependence are calibrated
on 2018 H2; 2019 is an untouched temporal test set. No test actual enters a
forecast, a residual distribution, or dependence estimation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from fusion.decision_agent import make_decisions
from fusion.forecast_contract import ForecastBundle, QUANTILE_COLUMNS
from fusion.fusion_agent import fuse_forecasts

LEVELS = np.arange(1, 100, dtype=float) / 100.0


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    stamp = pd.to_datetime(frame["timestamp_utc"], utc=True)
    local = stamp.dt.tz_convert("Europe/Berlin")
    hour = local.dt.hour.to_numpy()
    dow = local.dt.dayofweek.to_numpy()
    year_day = local.dt.dayofyear.to_numpy()
    solar_ratio = frame["solar_mw"] / frame["solar_capacity_mw"]
    features = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "year_sin": np.sin(2 * np.pi * year_day / 365.25),
        "year_cos": np.cos(2 * np.pi * year_day / 365.25),
        "load_lag25": frame["load_mw"].shift(25),
        "load_lag49": frame["load_mw"].shift(49),
        "load_lag169": frame["load_mw"].shift(169),
        "solar_ratio_lag25": solar_ratio.shift(25),
        "solar_ratio_lag49": solar_ratio.shift(49),
        "solar_ratio_lag169": solar_ratio.shift(169),
        "capacity_lag25": frame["solar_capacity_mw"].shift(25),
    })
    return features, local, solar_ratio


def _quantiles_by_hour(point: np.ndarray, calibration_point: np.ndarray,
                       calibration_actual: np.ndarray,
                       forecast_hour: np.ndarray,
                       calibration_hour: np.ndarray,
                       *, low: float = 0.0, high: float | np.ndarray = np.inf) -> np.ndarray:
    """Empirical, hour-conditioned residual quantiles from past validation only."""
    residual = calibration_actual - calibration_point
    result = np.empty((len(point), len(LEVELS)), dtype=float)
    for hour in range(24):
        selected = calibration_hour == hour
        if selected.sum() < 50:
            raise ValueError(f"too few calibration observations for local hour {hour}")
        offsets = np.quantile(residual[selected], LEVELS)
        result[forecast_hour == hour] = point[forecast_hour == hour, None] + offsets
    result = np.clip(result, low, high)
    if not np.isfinite(result).all() or (np.diff(result, axis=1) < -1e-9).any():
        raise ValueError("invalid or crossing marginal quantiles")
    return result


def _score(actual: np.ndarray, quantiles: np.ndarray) -> dict:
    error = actual[:, None] - quantiles
    pinball = np.maximum(LEVELS * error, (LEVELS - 1) * error).mean()
    return {
        "n_hours": int(len(actual)),
        "mean_pinball_loss_mw": float(pinball),
        "coverage_q05_q95": float(np.mean((actual >= quantiles[:, 4]) &
                                          (actual <= quantiles[:, 94]))),
        "mean_q05_q95_width_mw": float(np.mean(quantiles[:, 94] - quantiles[:, 4])),
        "median_mae_mw": float(np.mean(np.abs(actual - quantiles[:, 49]))),
        "quantile_crossings": int((np.diff(quantiles, axis=1) < -1e-9).sum()),
    }


def _forecast_frame(stamp: pd.Series, values: np.ndarray) -> pd.DataFrame:
    # ForecastBundle normalizes these timezone-naive UTC-equivalent values.
    return pd.DataFrame({"TIMESTAMP": stamp.dt.tz_localize(None).to_numpy(),
                         **{q: values[:, i] for i, q in enumerate(QUANTILE_COLUMNS)}})


def run(source: Path, output: Path, *, scenarios: int = 1000,
        seed: int = 2026) -> dict:
    source, output = source.resolve(), output.resolve()
    frame = pd.read_csv(source)
    required = {"timestamp_utc", "load_mw", "solar_mw", "solar_capacity_mw"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing cleaned OPSD columns: {sorted(required - set(frame))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("cleaned OPSD data contain missing values")
    features, local, solar_ratio = _features(frame)
    valid_rows = features.notna().all(axis=1)
    train = ((local >= pd.Timestamp("2017-01-01", tz="Europe/Berlin")) &
             (local < pd.Timestamp("2018-07-01", tz="Europe/Berlin")) & valid_rows).to_numpy()
    calibration = ((local >= pd.Timestamp("2018-07-01", tz="Europe/Berlin")) &
                   (local < pd.Timestamp("2019-01-01", tz="Europe/Berlin")) & valid_rows).to_numpy()
    test = ((local >= pd.Timestamp("2019-01-01", tz="Europe/Berlin")) &
            (local < pd.Timestamp("2020-01-01", tz="Europe/Berlin")) & valid_rows).to_numpy()
    if min(train.sum(), calibration.sum(), test.sum()) < 1000:
        raise ValueError("chronological train, calibration and test sets must each have >=1000 hours")
    if scenarios < 100:
        raise ValueError("use at least 100 Monte Carlo scenarios")

    load_columns = [c for c in features if not c.startswith("solar_") and c != "capacity_lag25"]
    solar_columns = [c for c in features if not c.startswith("load_")]
    model_args = dict(max_iter=150, max_leaf_nodes=31, learning_rate=0.06,
                      min_samples_leaf=30, l2_regularization=1.0, random_state=seed)
    load_model = HistGradientBoostingRegressor(**model_args)
    solar_model = HistGradientBoostingRegressor(**model_args)
    load_model.fit(features.loc[train, load_columns], frame.loc[train, "load_mw"])
    solar_model.fit(features.loc[train, solar_columns], solar_ratio.loc[train])

    cal_load_point = load_model.predict(features.loc[calibration, load_columns])
    test_load_point = load_model.predict(features.loc[test, load_columns])
    cal_solar_ratio_point = solar_model.predict(features.loc[calibration, solar_columns])
    test_solar_ratio_point = solar_model.predict(features.loc[test, solar_columns])
    cal_cap = features.loc[calibration, "capacity_lag25"].to_numpy()
    test_cap = features.loc[test, "capacity_lag25"].to_numpy()
    cal_hour = local.loc[calibration].dt.hour.to_numpy()
    test_hour = local.loc[test].dt.hour.to_numpy()
    cal_load_actual = frame.loc[calibration, "load_mw"].to_numpy()
    cal_solar_actual = frame.loc[calibration, "solar_mw"].to_numpy()

    load_cal_q = _quantiles_by_hour(cal_load_point, cal_load_point, cal_load_actual,
                                    cal_hour, cal_hour)
    solar_cal_q_ratio = _quantiles_by_hour(
        cal_solar_ratio_point, cal_solar_ratio_point, cal_solar_actual / cal_cap,
        cal_hour, cal_hour, high=1.0)
    load_test_q = _quantiles_by_hour(test_load_point, cal_load_point, cal_load_actual,
                                     test_hour, cal_hour)
    solar_test_q_ratio = _quantiles_by_hour(
        test_solar_ratio_point, cal_solar_ratio_point, cal_solar_actual / cal_cap,
        test_hour, cal_hour,
        high=1.0)
    solar_test_q = solar_test_q_ratio * test_cap[:, None]
    # Residual ranks are estimated exclusively on calibration observations.
    load_residual = cal_load_actual - load_cal_q[:, 49]
    solar_residual = cal_solar_actual - solar_cal_q_ratio[:, 49] * cal_cap
    rank_dependence = float(spearmanr(load_residual, solar_residual).statistic)
    if not np.isfinite(rank_dependence):
        raise ValueError("calibration dependence is not finite")
    rank_dependence = float(np.clip(rank_dependence, -0.99, 0.99))

    stamp = pd.to_datetime(frame.loc[test, "timestamp_utc"], utc=True).reset_index(drop=True)
    load_forecast = _forecast_frame(stamp, load_test_q)
    solar_forecast = _forecast_frame(stamp, solar_test_q)
    load_bundle = ForecastBundle("OPSD-DE load baseline", load_forecast, "MW", "DE", "24h",
                                 physical_min=0.0)
    solar_bundle = ForecastBundle("OPSD-DE solar baseline", solar_forecast, "MW", "DE", "24h",
                                  metadata={"capacity_mw": float(frame.loc[train | calibration,
                                                                            "solar_capacity_mw"].max())},
                                  physical_min=0.0,
                                  physical_max=float(frame.loc[train | calibration,
                                                                 "solar_capacity_mw"].max()))
    output.mkdir(parents=True, exist_ok=True)
    load_forecast.to_csv(output / "load_q01_q99_2019.csv", index=False)
    solar_forecast.to_csv(output / "solar_q01_q99_2019.csv", index=False)
    actual_load = frame.loc[test, "load_mw"].to_numpy()
    actual_solar = frame.loc[test, "solar_mw"].to_numpy()
    actual_net = actual_load - actual_solar
    pd.DataFrame({"TIMESTAMP": load_forecast["TIMESTAMP"], "load_actual_mw": actual_load,
                  "solar_actual_mw": actual_solar, "net_load_actual_mw": actual_net,
                  "capacity_lag25_mw": test_cap}).to_csv(output / "actuals_2019.csv", index=False)

    methods: dict[str, dict] = {}
    for name, dependence in (("copula", rank_dependence), ("independent", 0.0)):
        monthly_forecasts = []
        month_reports = []
        months = local.loc[test].dt.month.to_numpy()
        for month in range(1, 13):
            take = months == month
            load_part = load_bundle.replace_frame(load_forecast.loc[take].reset_index(drop=True))
            solar_part = solar_bundle.replace_frame(solar_forecast.loc[take].reset_index(drop=True))
            fused = fuse_forecasts(load_part, solar_part, n_scenarios=scenarios,
                                   seed=seed + month, dependence=dependence,
                                   output_location="DE")
            decisions = make_decisions(fused)
            monthly_forecasts.append(fused.forecast)
            month_reports.append({"month": month, "summary": decisions.summary,
                                  "risk_flags": decisions.risk_flags,
                                  "recommendations": decisions.recommendations})
        combined = pd.concat(monthly_forecasts, ignore_index=True)
        if not combined["TIMESTAMP"].reset_index(drop=True).equals(
                load_forecast["TIMESTAMP"].reset_index(drop=True)):
            raise ValueError("monthly fusion does not match test timestamps")
        combined.to_csv(output / f"net_load_{name}_q01_q99_2019.csv", index=False)
        (output / f"decisions_{name}_2019.json").write_text(
            json.dumps(month_reports, indent=2) + "\n", encoding="utf-8")
        methods[name] = _score(actual_net, combined[list(QUANTILE_COLUMNS)].to_numpy())

    report = {
        "experiment": "measured_OPSD_DE_hourly_simple_day_ahead_baseline",
        "not_existing_GEFCom_forecasting_agents": True,
        "source_path": str(source), "source_sha256": _hash(source),
        "unit": "MW", "area": "DE", "timestamp": "UTC interval start",
        "train": {"period": "2017-01 to 2018-06 local time", "n_hours": int(train.sum())},
        "calibration": {"period": "2018-07 to 2018-12 local time", "n_hours": int(calibration.sum())},
        "test": {"period": "2019-01 to 2019-12 local time", "n_hours": int(test.sum())},
        "horizon": "strict 24h-ahead from hourly interval start; no measured feature uses an interval later than t-25h",
        "model": "separate HistGradientBoostingRegressor point models with hour-conditioned empirical residual quantiles",
        "weather_features": False,
        "dependence_calibration": "Spearman rank correlation of 2018 H2 marginal median residuals only",
        "calibration_rank_correlation": rank_dependence,
        "n_scenarios_per_hour": scenarios, "seed": seed,
        "load": _score(actual_load, load_test_q),
        "solar": _score(actual_solar, solar_test_q),
        "net_load": methods,
        "limitations": [
            "Simple calendar/lag baseline; does not retrain the existing GEFCom agent architectures.",
            "No weather forecasts or operator action labels are used.",
            "Reserve and ramp outputs are transparent research proxies, not dispatch instructions.",
            "OPSD total-load and solar generation definitions require operational reconciliation before interpreting load minus solar as conventional dispatch.",
        ],
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output,
                         scenarios=args.scenarios, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

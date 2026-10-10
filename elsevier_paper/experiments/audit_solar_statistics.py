"""Journal-only, synchronized-zone inference from archived solar forecasts.

Does not retrain models or overwrite the original pipeline evaluation.
Run: python -m elsevier_paper.experiments.audit_solar_statistics
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
SOLAR = ROOT / "agentic_solar/solar-forecasting-GEFCom2014"
OUTPUT = Path(__file__).resolve().parent / "solar_statistical_audit"
MODELS = ("Ensemble", "LSTM", "TCN", "XGBoost", "LightGBM", "CatBoost",
          "TFT", "iTransformer", "PatchTST")
QCOLS = [f"q{i:02d}" for i in range(1, 100)]
LEVELS = np.arange(1, 100) / 100.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hourly_zone_mean(index: pd.DataFrame, losses: np.ndarray) -> np.ndarray:
    """Aggregate simultaneous zones before serial-dependence inference."""
    panel = index.assign(loss=np.asarray(losses, dtype=float)).pivot(
        index="TIMESTAMP", columns="ZONEID", values="loss")
    if panel.shape != (720, 3) or panel.isna().any().any():
        raise ValueError("expected 720 common hours and three complete solar zones")
    return panel.mean(axis=1).to_numpy()


def hac_mean(difference: np.ndarray, lag: int) -> dict:
    d = np.asarray(difference, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("HAC requires a finite one-dimensional paired loss series")
    if not 0 <= lag < len(d):
        raise ValueError("HAC lag must be nonnegative and shorter than the series")
    centered = d - d.mean()
    variance = np.dot(centered, centered) / len(d)
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * np.dot(centered[k:], centered[:-k]) / len(d)
    se = float(np.sqrt(max(float(variance), 0) / len(d)))
    statistic = float(d.mean() / se) if se else (0.0 if d.mean() == 0 else float(np.sign(d.mean()) * np.inf))
    return {"mean_difference": float(d.mean()), "standard_error": se,
            "statistic": statistic, "p_value": float(2 * norm.sf(abs(statistic))),
            "ci95_low": float(d.mean() - norm.ppf(.975) * se),
            "ci95_high": float(d.mean() + norm.ppf(.975) * se), "lag": lag}


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid p-values")
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def shared_day_ci(hourly_losses: np.ndarray, *, seed: int = 2026,
                  draws: int = 5000) -> tuple[float, float]:
    values = np.asarray(hourly_losses, dtype=float)
    if values.shape != (720,) or not np.isfinite(values).all():
        raise ValueError("expected a complete 30-day hourly series")
    # Hourly losses already average simultaneous zones. Each draw retains
    # every hour and all zones of each selected competition day together.
    days = values.reshape(30, 24).mean(axis=1)
    averages = np.random.default_rng(seed).choice(days, size=(draws, 30)).mean(axis=1)
    return tuple(float(x) for x in np.quantile(averages, [.025, .975]))


def run(output: Path = OUTPUT, keyed_predictions: Path | None = None) -> dict:
    outputs = SOLAR / "pipeline/outputs"
    target = outputs / "splits/XGBoost/test.csv"
    actual = pd.read_csv(target)
    actual["TIMESTAMP"] = pd.to_datetime(actual.TIMESTAMP)
    actual = actual.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
    index = actual[["ZONEID", "TIMESTAMP"]]
    y = actual.POWER.to_numpy()
    if index.duplicated().any():
        raise ValueError("duplicate zone-hour targets")
    output.mkdir(parents=True, exist_ok=True)
    records, hourly, metrics = {}, {}, []
    for model in MODELS:
        path = (keyed_predictions / model / "test_keyed_q01_q99.csv" if keyed_predictions
                else outputs / f"predictions/{model}_predictions.csv")
        frame = pd.read_csv(path)
        if set(QCOLS) - set(frame) or len(frame) != len(actual):
            raise ValueError(f"{model} prediction shape mismatch")
        if keyed_predictions:
            frame.TIMESTAMP = pd.to_datetime(frame.TIMESTAMP)
            frame = frame.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
            if not frame[["ZONEID", "TIMESTAMP"]].equals(index):
                raise ValueError(f"{model} recorded prediction keys differ")
            np.testing.assert_allclose(frame.POWER, y, rtol=0, atol=1e-12)
        split = outputs / f"splits/{model if model != 'Ensemble' else 'XGBoost'}/test.csv"
        peer = pd.read_csv(split, usecols=["ZONEID", "TIMESTAMP", "POWER"])
        peer.TIMESTAMP = pd.to_datetime(peer.TIMESTAMP)
        peer = peer.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
        if not peer[["ZONEID", "TIMESTAMP"]].equals(index):
            raise ValueError(f"{model} split zone-hour keys differ")
        np.testing.assert_allclose(peer.POWER, y, rtol=0, atol=1e-12)
        q = np.sort(np.clip(frame[QCOLS].to_numpy(dtype=np.float32), 0, 1), axis=1)
        _record(model, q, path, split, index, y, output, records, hourly, metrics)
    path = SOLAR / "data/GEFCom2014-S_V2/Solar/Task 15/benchmark15.csv"
    benchmark = pd.read_csv(path)
    benchmark.TIMESTAMP = pd.to_datetime(benchmark.TIMESTAMP, format="%Y%m%d %H:%M")
    benchmark = benchmark.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
    if not benchmark[["ZONEID", "TIMESTAMP"]].equals(index):
        raise ValueError("released benchmark zone-hour alignment fails")
    q = np.sort(np.clip(benchmark.iloc[:, 2:].to_numpy(dtype=np.float32), 0, 1), axis=1)
    _record("Released point benchmark", q, path, target, index, y, output,
            records, hourly, metrics)
    comparisons = []
    for model in (*MODELS[1:], "Released point benchmark"):
        difference = hourly[model] - hourly["Ensemble"]
        comparisons.append({"comparator": model, **hac_mean(difference, 24),
                            "sensitivity_lag_168": hac_mean(difference, 168),
                            "shared_day_bootstrap_ci95": shared_day_ci(difference)})
    for row, adjusted in zip(comparisons, holm_adjust([r["p_value"] for r in comparisons])):
        row["holm_p_value"] = float(adjusted)
    report = {"protocol": "exploratory retrospective journal audit; no model reselection",
              "aggregation": "equal-weight mean of three zones at each common timestamp",
              "n_zone_hours": 2160, "n_common_hours": 720, "n_competition_days": 30,
              "primary_hac_lag_hours": 24, "sensitivity_hac_lag_hours": 168,
              "bootstrap_draws": 5000, "bootstrap_seed": 2026,
              "bootstrap_limit": "shared-day bootstrap preserves within-day/cross-zone dependence, not dependence between days",
              "comparison_family": "nine ensemble comparisons; Holm correction for primary p-values",
              "benchmark_all_quantiles_identical": bool(np.all(np.ptp(q, axis=1) == 0)),
              "target_sha256": sha(target), "artifacts": records,
              "comparisons": comparisons,
              "row_order_limit": ("CPU checkpoint replays include checked original output zone-hour keys; historical CSV identity is not inferred"
                                  if keyed_predictions else "prediction CSVs lack original keys; sidecars document reconstructed expected writer order, not independently recorded prediction identities")}
    pd.DataFrame(metrics).to_csv(output / "solar_metrics_with_shared_day_ci.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, tuple))}
                  for r in comparisons]).to_csv(output / "solar_hac_comparisons.csv", index=False)
    (output / "statistical_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _record(model, q, path, split, index, y, output, records, hourly, metrics):
    e = y[:, None] - q
    losses = np.maximum(LEVELS * e, (LEVELS - 1) * e).mean(axis=1)
    hourly[model] = hourly_zone_mean(index, losses)
    lo, hi = shared_day_ci(hourly[model])
    day = y > .001  # target-based subgroup, not astronomical daylight
    metric = {"model": model, "pinball": float(losses.mean()), "pinball_ci95_low": lo,
              "pinball_ci95_high": hi, "mae": float(np.abs(y - q[:, 49]).mean()),
              "coverage_q10_q90": float(((y >= q[:, 9]) & (y <= q[:, 89])).mean()),
              "width_q10_q90": float((q[:, 89] - q[:, 9]).mean()),
              "positive_generation_rows": int(day.sum()),
              "positive_generation_pinball": float(losses[day].mean()),
              "positive_generation_coverage": float(((y[day] >= q[day, 9]) & (y[day] <= q[day, 89])).mean())}
    sidecar = output / f"{model.replace(' ', '_')}_expected_index.csv"
    index.to_csv(sidecar, index=False)
    records[model] = {"prediction_path": str(path.relative_to(ROOT)), "prediction_sha256": sha(path),
                      "split_path": str(split.relative_to(ROOT)), "split_sha256": sha(split),
                      "expected_index_sha256": sha(sidecar), "row_count": len(y)}
    metrics.append(metric)


if __name__ == "__main__":
    result = run()
    print(f"Verified {len(result['artifacts'])} artifacts; wrote {OUTPUT}")
    for comparison in result["comparisons"]:
        print(comparison["comparator"], "p:", comparison["p_value"], "Holm p:", comparison["holm_p_value"])

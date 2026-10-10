"""Independently verify the forecasting metrics used in the journal draft.

Run from the repository root:

    agentic/.venv/bin/python -m elsevier_paper.experiments.verify_forecasts

This audit reads prediction and target artifacts but does not retrain models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "forecast_artifact_audit.json"
QCOLS = [f"q{i:02d}" for i in range(1, 100)]
LEVELS = np.arange(1, 100, dtype=float) / 100.0


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(y: np.ndarray, q: np.ndarray) -> dict:
    error = y[:, None] - q
    loss = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1) * error)
    return {
        "n": len(y),
        "pinball": float(loss.mean()),
        "mae": float(np.mean(np.abs(y - q[:, 49]))),
        "rmse": float(np.sqrt(np.mean((y - q[:, 49]) ** 2))),
        "coverage_q10_q90": float(np.mean((y >= q[:, 9]) & (y <= q[:, 89]))),
        "sharpness_q10_q90": float(np.mean(q[:, 89] - q[:, 9])),
        "crossing_pairs": int(np.sum(np.diff(q, axis=1) < -1e-10)),
    }


def audit_load() -> dict:
    root = ROOT / "agentic"
    expected = {
        "GBR": {"pinball": 11.082142, "mae": 30.6426, "coverage_q10_q90": 0.734},
        "Linear": {"pinball": 21.515278, "mae": 50.1339, "coverage_q10_q90": 0.437},
    }
    canonical = pd.read_csv(root / "pipeline/outputs/splits/Linear/test.csv",
                            usecols=["TIMESTAMP", "LOAD"])
    records = {}
    for model in expected:
        path = root / f"outputs/test_predictions_monotone/{model}_predictions_monotone.csv"
        frame = pd.read_csv(path)
        if set(QCOLS) - set(frame.columns):
            raise ValueError(f"{model} load predictions miss quantiles")
        if len(frame) != len(canonical) or not frame["TIMESTAMP"].equals(canonical["TIMESTAMP"]):
            raise ValueError(f"{model} load timestamp alignment fails")
        y = canonical["LOAD"].to_numpy(float)
        np.testing.assert_allclose(y, frame["actual_LOAD"].to_numpy(float), atol=1e-9)
        measured = _metrics(y, frame[QCOLS].to_numpy(float))
        np.testing.assert_allclose(measured["pinball"], expected[model]["pinball"], atol=1e-5)
        np.testing.assert_allclose(measured["mae"], expected[model]["mae"], atol=1e-4)
        np.testing.assert_allclose(measured["coverage_q10_q90"],
                                   expected[model]["coverage_q10_q90"], atol=5e-4)
        records[model] = {**measured, "sha256": _sha(path)}
    return {
        "source": "agentic current two-model monotone-corrected run",
        "target_sha256": _sha(root / "pipeline/outputs/splits/Linear/test.csv"),
        "models": records,
        "unit": "load data unit as supplied by GEFCom track; verify MW label against source",
    }


def audit_solar() -> dict:
    root = ROOT / "agentic_solar/solar-forecasting-GEFCom2014/pipeline/outputs"
    target_path = root / "splits/XGBoost/test.csv"
    actual = pd.read_csv(target_path).sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
    y = actual["POWER"].to_numpy(float)
    reported = pd.read_csv(root / "research_evaluation/research_results.csv")
    records = {}
    for _, row in reported[reported["Model"] != "Official benchmark"].iterrows():
        model = row["Model"]
        path = root / f"predictions/{model}_predictions.csv"
        frame = pd.read_csv(path)
        if set(QCOLS) - set(frame.columns) or len(frame) != len(y):
            raise ValueError(f"{model} solar shape or quantile columns fail")
        # The research-evaluation implementation uses float32, clips physical
        # capacity fractions to [0,1], then monotonically rearranges each row.
        q = np.sort(np.clip(frame[QCOLS].to_numpy(dtype=np.float32), 0.0, 1.0), axis=1)
        measured = _metrics(y, q)
        for key, reported_key in (
            ("pinball", "Pinball_Loss"), ("mae", "MAE"),
            ("rmse", "RMSE"), ("coverage_q10_q90", "Coverage_80%"),
            ("sharpness_q10_q90", "Sharpness"),
        ):
            np.testing.assert_allclose(measured[key], row[reported_key], atol=2e-6)
        records[model] = {**measured, "sha256": _sha(path)}
    benchmark_path = ROOT / ("agentic_solar/solar-forecasting-GEFCom2014/data/"
                             "GEFCom2014-S_V2/Solar/Task 15/benchmark15.csv")
    benchmark = pd.read_csv(benchmark_path)
    benchmark = benchmark.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
    benchmark_q = np.sort(np.clip(benchmark.iloc[:, 2:].to_numpy(dtype=np.float32),
                                  0.0, 1.0), axis=1)
    if benchmark_q.shape != (len(y), 99):
        raise ValueError("official solar benchmark shape mismatch")
    benchmark_metrics = _metrics(y, benchmark_q)
    official = reported.loc[reported["Model"] == "Official benchmark"].iloc[0]
    np.testing.assert_allclose(benchmark_metrics["pinball"], official["Pinball_Loss"],
                               atol=2e-6)
    records["Official benchmark"] = {**benchmark_metrics, "sha256": _sha(benchmark_path)}
    if len(y) != 2160 or len(records) != 10:
        raise ValueError("unexpected solar test size or model count")
    return {
        "source": "solar nine-model research evaluation, June 2014, three zones",
        "target_sha256": _sha(target_path),
        "unit": "fraction of installed capacity, not MW",
        "models": records,
    }


def audit_legacy_fit_source() -> dict:
    path = ROOT / "coding/outputs/results_summary.csv"
    table = pd.read_csv(path)
    xgb = ROOT / "coding/outputs/predictions/XGBoost_predictions.csv"
    columns = pd.read_csv(xgb, nrows=0).columns
    return {
        "source": "legacy coding artifacts associated with submitted FIT comparison",
        "results_sha256": _sha(path),
        "model_count": len(table),
        "different_test_counts": sorted(table["n_test"].unique().tolist()),
        "xgboost_missing_quantiles": sorted(set(QCOLS) - set(columns)),
        "legacy_patchtst_pinball": float(table.loc[table["Model"] == "PatchTST", "Pinball_Loss"].iloc[0]),
        "not_used_as_new_journal_evidence": True,
    }


def main() -> int:
    audit = {"load": audit_load(), "solar": audit_solar(),
             "legacy_fit_source": audit_legacy_fit_source()}
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Verified {len(audit['load']['models'])} load artifacts, "
          f"{len(audit['solar']['models']) - 1} solar model artifacts, and the "
          f"official solar benchmark; wrote {OUT}")
    print("Legacy FIT-source quantile gaps:",
          audit["legacy_fit_source"]["xgboost_missing_quantiles"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

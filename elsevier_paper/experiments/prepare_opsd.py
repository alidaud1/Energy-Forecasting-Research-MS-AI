"""Validate the official OPSD Germany hourly subset for measured experiments.

Input is the CSV returned by OPSD's resource-3 custom-download form with
Region=DE, the three named variables below, and local dates 2017-01-01 through
2019-12-31. No load or generation observation is imputed. The 24 missing
capacity entries on the final day are carried forward from the last published
capacity, flagged in the output, and never used as target observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/"
LOAD = "DE_load_actual_entsoe_transparency"
SOLAR = "DE_solar_generation_actual"
CAPACITY = "DE_solar_capacity"
REQUIRED = ("utc_timestamp", "cet_cest_timestamp", LOAD, SOLAR, CAPACITY)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source: Path, output: Path, *, expected_rows: int = 26280) -> dict:
    """Fail closed on incomplete/misaligned data; write a normalized UTC CSV."""
    source = source.resolve()
    output = output.resolve()
    if source.suffix == ".partial":
        raise ValueError("incomplete .partial downloads cannot be prepared")
    raw = pd.read_csv(source)
    missing_columns = sorted(set(REQUIRED) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"missing OPSD columns: {missing_columns}")
    if len(raw) != expected_rows:
        raise ValueError(f"expected {expected_rows} hourly rows, found {len(raw)}")

    stamp = pd.to_datetime(raw["utc_timestamp"], utc=True, errors="raise")
    if stamp.isna().any() or stamp.duplicated().any():
        raise ValueError("UTC timestamps are missing or duplicated")
    if not stamp.is_monotonic_increasing or not stamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("UTC timestamps are not strictly consecutive hours")
    if expected_rows == 26280 and (
        stamp.iloc[0] != pd.Timestamp("2016-12-31T23:00:00Z")
        or stamp.iloc[-1] != pd.Timestamp("2019-12-31T22:00:00Z")
    ):
        raise ValueError("source is not the complete 2017–2019 local-date window")

    numeric = raw[[LOAD, SOLAR, CAPACITY]].apply(pd.to_numeric, errors="raise")
    if numeric[[LOAD, SOLAR]].isna().any().any():
        raise ValueError("actual load or solar generation has missing observations")
    if not np.isfinite(numeric[[LOAD, SOLAR]].to_numpy()).all():
        raise ValueError("actual load or solar generation has non-finite values")
    if (numeric[[LOAD, SOLAR]] < 0).any().any():
        raise ValueError("actual load or solar generation is negative")

    capacity_missing = numeric[CAPACITY].isna()
    capacity = numeric[CAPACITY].ffill(limit=24)
    if capacity.isna().any() or not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError("capacity has an unresolvable gap or invalid value")
    if (numeric[SOLAR] > capacity).any():
        raise ValueError("solar generation exceeds reported capacity")

    cleaned = pd.DataFrame({
        "timestamp_utc": stamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "load_mw": numeric[LOAD],
        "solar_mw": numeric[SOLAR],
        "solar_capacity_mw": capacity,
        "capacity_forward_filled": capacity_missing,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    report = {
        "source_url": SOURCE_URL,
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "output_path": str(output),
        "output_sha256": _sha256(output),
        "area": "DE",
        "frequency": "1h",
        "unit": "MW",
        "timestamp_basis": "UTC interval start",
        "rows": len(cleaned),
        "first_utc": cleaned["timestamp_utc"].iloc[0],
        "last_utc": cleaned["timestamp_utc"].iloc[-1],
        "load_missing": int(numeric[LOAD].isna().sum()),
        "solar_missing": int(numeric[SOLAR].isna().sum()),
        "capacity_forward_filled_hours": int(capacity_missing.sum()),
        "capacity_fill_method": "last observed capacity, at most 24 hours; never fill load or solar",
        "load_range_mw": [float(numeric[LOAD].min()), float(numeric[LOAD].max())],
        "solar_range_mw": [float(numeric[SOLAR].min()), float(numeric[SOLAR].max())],
        "net_load_range_mw": [float((numeric[LOAD] - numeric[SOLAR]).min()),
                              float((numeric[LOAD] - numeric[SOLAR]).max())],
        "no_forecast_results": True,
    }
    manifest = output.with_name(output.stem + "_manifest.json")
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

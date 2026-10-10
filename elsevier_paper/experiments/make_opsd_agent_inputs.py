"""Create isolated, strict-24-hour inputs for porting the forecasting agents.

This does not invoke or overwrite either GEFCom pipeline. It provides one
matching train/calibration/test protocol and explicitly excludes unmeasured
weather and fabricated GEFCom zone identifiers. A model port may consume
these tables only after its own feature/scaler/checkpoint interface is adapted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_opsd_baseline import _features

CALENDAR = ("hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "year_sin", "year_cos")
LOAD_FEATURES = CALENDAR + ("load_lag25", "load_lag49", "load_lag169")
SOLAR_FEATURES = CALENDAR + ("solar_ratio_lag25", "solar_ratio_lag49",
                              "solar_ratio_lag169", "capacity_lag25")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    data = pd.read_csv(source)
    required = {"timestamp_utc", "load_mw", "solar_mw", "solar_capacity_mw"}
    if not required.issubset(data.columns):
        raise ValueError(f"missing measured columns: {sorted(required - set(data))}")
    if len(data) != 26280 or data[list(required)].isna().any().any():
        raise ValueError("expected the complete 26,280-hour cleaned OPSD-DE series")
    stamp = pd.to_datetime(data["timestamp_utc"], utc=True)
    if stamp.duplicated().any() or not stamp.is_monotonic_increasing or not (
            stamp.diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("measured UTC timestamps are not consecutive hours")

    features, local, _ = _features(data)
    available = features[list(dict.fromkeys(LOAD_FEATURES + SOLAR_FEATURES))].notna().all(axis=1)
    boundaries = {
        "train": ("2017-01-01", "2018-07-01"),
        "calibration": ("2018-07-01", "2019-01-01"),
        "test": ("2019-01-01", "2020-01-01"),
    }
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    files = {}
    for split, (start, end) in boundaries.items():
        mask = ((local >= pd.Timestamp(start, tz="Europe/Berlin")) &
                (local < pd.Timestamp(end, tz="Europe/Berlin")) & available).to_numpy()
        if mask.sum() < 1000:
            raise ValueError(f"{split} has too few usable hourly rows")
        timestamps = stamp.loc[mask].dt.strftime("%Y-%m-%dT%H:%M:%SZ").reset_index(drop=True)
        load = pd.DataFrame({"TIMESTAMP": timestamps, "AREA": "DE",
                             "LOAD": data.loc[mask, "load_mw"].to_numpy()})
        solar = pd.DataFrame({"TIMESTAMP": timestamps, "AREA": "DE",
                              "POWER": (data.loc[mask, "solar_mw"].to_numpy() /
                                        features.loc[mask, "capacity_lag25"].to_numpy())})
        for name in LOAD_FEATURES:
            load[name] = features.loc[mask, name].to_numpy()
        for name in SOLAR_FEATURES:
            solar[name] = features.loc[mask, name].to_numpy()
        if load.isna().any().any() or solar.isna().any().any():
            raise ValueError(f"{split} has missing target or model features")
        if (load["LOAD"] < 0).any() or not solar["POWER"].between(0, 1).all():
            raise ValueError(f"{split} target is outside its physical bounds")
        if not load["TIMESTAMP"].equals(solar["TIMESTAMP"]):
            raise ValueError(f"{split} load/solar timestamps do not align")
        counts[split] = len(load)
        for domain, table in (("load", load), ("solar", solar)):
            path = output / f"{domain}_{split}.csv"
            table.to_csv(path, index=False)
            files[path.name] = _hash(path)

    manifest = {
        "source": str(source), "source_sha256": _hash(source),
        "area": "DE", "timestamp": "UTC interval start",
        "forecast_issue": "strictly 24 hours before target interval start",
        "latest_measured_feature": "target interval minus 25 hours",
        "splits_local_calendar": boundaries,
        "split_rows": counts,
        "load_target": "LOAD, MW",
        "solar_target": "POWER = measured solar MW / capacity observed 25 hours earlier; ratio in [0,1]",
        "load_features": list(LOAD_FEATURES),
        "solar_features": list(SOLAR_FEATURES),
        "weather_forecasts_available": False,
        "not_native_GEFCom_pipeline_inputs": True,
        "no_fabricated_zone_ids": True,
        "model_training_completed": False,
        "file_sha256": files,
        "caveat": "Port native model architectures with isolated scalers/checkpoints; do not run GEFCom-specific preprocessors or overwrite their outputs.",
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(args.source, args.output)
    print(json.dumps({"split_rows": result["split_rows"],
                      "output": str(args.output.resolve()),
                      "model_training_completed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

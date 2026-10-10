"""Prepare an independently timed 2020 German-grid holdout.

The OPSD 2020 custom export has no solar-capacity observations.  Only the
last *published* 2019 capacity is carried forward as an input proxy; load
and solar generation observations are never imputed.  The proxy is disclosed
in the manifest and must not be described as measured 2020 capacity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .make_opsd_agent_inputs import LOAD_FEATURES, SOLAR_FEATURES, _hash
from .run_opsd_baseline import _features
from .prepare_opsd import LOAD, SOLAR, CAPACITY


def prepare(history_path: Path, source_path: Path, output: Path) -> dict:
    history_path, source_path, output = (p.resolve() for p in
                                         (history_path, source_path, output))
    history = pd.read_csv(history_path)
    source = pd.read_csv(source_path)
    needed = {"utc_timestamp", "cet_cest_timestamp", LOAD, SOLAR, CAPACITY}
    if not needed.issubset(source):
        raise ValueError(f"missing OPSD fields: {sorted(needed - set(source))}")
    stamp = pd.to_datetime(source.utc_timestamp, utc=True)
    if len(source) != 4367 or stamp.iloc[0] != pd.Timestamp("2019-12-31T23:00:00Z") or \
            stamp.iloc[-1] != pd.Timestamp("2020-06-30T21:00:00Z") or \
            not stamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("2020 export is not the complete Jan–Jun local-date interval")
    if source[[LOAD, SOLAR]].isna().any().any() or \
            not np.isfinite(source[[LOAD, SOLAR]].to_numpy(float)).all() or \
            (source[[LOAD, SOLAR]] < 0).any().any():
        raise ValueError("2020 measured load or solar is invalid")
    if source[CAPACITY].notna().any():
        raise ValueError("capacity-proxy protocol expects no 2020 capacity observations")
    old_stamp = pd.to_datetime(history.timestamp_utc, utc=True)
    if old_stamp.iloc[-1] + pd.Timedelta(hours=1) != stamp.iloc[0]:
        raise ValueError("historical and holdout UTC hours are not adjacent")
    last_capacity = float(history.solar_capacity_mw.iloc[-1])
    if last_capacity <= 0 or source[SOLAR].max() > last_capacity:
        raise ValueError("last-known solar capacity is not a plausible upper bound")
    continuation = pd.DataFrame({
        "timestamp_utc": stamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "load_mw": source[LOAD], "solar_mw": source[SOLAR],
        "solar_capacity_mw": last_capacity,
    })
    joined = pd.concat([history.drop(columns=["capacity_forward_filled"], errors="ignore"),
                        continuation], ignore_index=True)
    features, local, _ = _features(joined)
    take = ((local >= pd.Timestamp("2020-01-01", tz="Europe/Berlin")) &
            (local < pd.Timestamp("2020-07-01", tz="Europe/Berlin")))
    if int(take.sum()) != 4367 or features.loc[take].isna().any().any():
        raise ValueError("2020 features are incomplete")
    chosen = joined.loc[take].reset_index(drop=True)
    feat = features.loc[take].reset_index(drop=True)
    timestamps = chosen.timestamp_utc
    load = pd.DataFrame({"TIMESTAMP": timestamps, "AREA": "DE", "LOAD": chosen.load_mw})
    solar = pd.DataFrame({"TIMESTAMP": timestamps, "AREA": "DE",
                          "POWER": chosen.solar_mw / feat.capacity_lag25})
    for name in LOAD_FEATURES:
        load[name] = feat[name]
    for name in SOLAR_FEATURES:
        solar[name] = feat[name]
    if not solar.POWER.between(0, 1).all():
        raise ValueError("2020 solar ratio exceeds capacity proxy")
    output.mkdir(parents=True, exist_ok=True)
    load.to_csv(output / "load_test_2020.csv", index=False)
    solar.to_csv(output / "solar_test_2020.csv", index=False)
    actual = pd.DataFrame({"TIMESTAMP": timestamps,
                           "load_actual_mw": chosen.load_mw,
                           "solar_actual_mw": chosen.solar_mw,
                           "net_load_actual_mw": chosen.load_mw - chosen.solar_mw,
                           "capacity_proxy_mw": last_capacity})
    actual.to_csv(output / "actuals_2020.csv", index=False)
    report = {"source_url": "https://data.open-power-system-data.org/time_series/2020-10-06/",
              "source_sha256": _hash(source_path), "history_sha256": _hash(history_path),
              "test_hours": len(load), "first_utc": timestamps.iloc[0],
              "last_utc": timestamps.iloc[-1],
              "capacity_proxy_mw": last_capacity,
              "capacity_status": "All 2020 capacity values absent; final published 2019 value held fixed as a forecast input proxy, not a measured 2020 capacity.",
              "max_measured_solar_mw": float(chosen.solar_mw.max()),
              "load_features": list(LOAD_FEATURES), "solar_features": list(SOLAR_FEATURES),
              "files_sha256": {name: _hash(output / name) for name in
                               ("load_test_2020.csv", "solar_test_2020.csv", "actuals_2020.csv")}}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.history, args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

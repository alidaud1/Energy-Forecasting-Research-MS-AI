"""Adapters from the existing load and solar prediction artifacts.

These functions deliberately sit outside both forecasting repositories.  They
make the integration boundary explicit while leaving each pipeline's internal
files and state untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .forecast_contract import ForecastBundle, ForecastContractError, QUANTILE_COLUMNS

# The solar track reports output as a fraction of each zone's installed
# capacity in [0, 1]; GEFCom2014-S never publishes the capacity in MW.  Net load
# is a difference of powers, so a fraction cannot be subtracted from a MW demand
# series -- doing so silently contributes ~0.06% and leaves net load equal to
# load.  Callers must therefore state the capacity they are assuming, and it is
# recorded in the manifest as an assumption rather than a measurement.
SOLAR_ACTUAL_COLUMN = "POWER"


def _read(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ForecastContractError(f"file does not exist: {resolved}")
    return pd.read_csv(resolved)


def load_bundle(path: str | Path, *, location: str = "load-grid",
                horizon: str = "24h") -> ForecastBundle:
    """Adapt an agentic-load prediction CSV (TIMESTAMP, actual_LOAD, q01..q99)."""
    return ForecastBundle.from_csv(path, source="load", unit="MW",
                                   location=location, horizon=horizon,
                                   actual_column="actual_LOAD",
                                   metadata={"prediction_file": str(Path(path).resolve())},
                                   physical_min=0.0)


def solar_bundle(path: str | Path, *, timestamps_path: str | Path,
                 capacity_mw: float,
                 location: str | None = None, horizon: str = "24h",
                 timestamp_column: str = "TIMESTAMP",
                 location_column: str | None = "ZONEID",
                 location_value: object | None = 1,
                 capacity_basis: str | None = None) -> ForecastBundle:
    """Adapt the existing solar prediction CSV, which stores quantiles only.

    Solar model artifacts omit timestamps, so the caller supplies the aligned
    index artifact -- either the GEFCom benchmark file or the solar pipeline's
    own ``splits/<Model>/test.csv``.  Both list the same (zone, timestamp)
    sequence in the same order as the predictions, and the row counts are
    checked before the two are zipped; silent synthetic timestamps and silent
    truncation are both refused.  When the index artifact carries the ``POWER``
    actuals column it is attached, which is what lets fusion estimate load/solar
    dependence instead of assuming independence.

    ``capacity_mw`` converts the published [0, 1] capacity fraction into MW and
    is mandatory: there is no defensible default, and the previous code's
    implicit "these numbers are already MW" made solar a rounding error against
    demand.
    """
    if not np.isfinite(capacity_mw) or capacity_mw <= 0:
        raise ForecastContractError("capacity_mw must be a positive, finite number of MW")
    prediction_path, index_path = Path(path).resolve(), Path(timestamps_path).resolve()
    frame = _read(prediction_path)
    missing = [q for q in QUANTILE_COLUMNS if q not in frame.columns]
    if missing:
        raise ForecastContractError(f"solar predictions missing quantiles: {missing[:5]}")
    fractions = frame.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float)
    if np.nanmax(fractions) > 1.5:
        raise ForecastContractError(
            "solar predictions exceed 1.5; they do not look like capacity fractions, "
            "so applying capacity_mw would double-count the conversion")

    index = _read(index_path)
    if timestamp_column not in index.columns:
        raise ForecastContractError(f"index file lacks {timestamp_column!r}")
    if len(index) != len(frame):
        raise ForecastContractError(
            f"index rows ({len(index)}) do not match prediction rows ({len(frame)}); "
            "the two artifacts are not row-aligned")
    timestamps = pd.to_datetime(index[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ForecastContractError(f"index file has unparseable timestamps in {timestamp_column!r}")

    # Multi-zone artifacts are stored zone-major in the same order as the index,
    # so positional selection is what pairs a prediction row with its zone.
    keep = np.ones(len(frame), dtype=bool)
    if location_column is not None:
        if location_column not in index.columns:
            raise ForecastContractError(f"index file lacks {location_column!r}")
        keep = (index[location_column] == location_value).to_numpy()
        if not keep.any():
            raise ForecastContractError(
                f"no rows for {location_column}={location_value!r} in {index_path.name}")
    selected = frame.loc[keep].reset_index(drop=True)
    stamps = timestamps.loc[keep].reset_index(drop=True)

    columns = {"TIMESTAMP": stamps}
    scaled = selected.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float) * capacity_mw
    columns.update({q: scaled[:, i] for i, q in enumerate(QUANTILE_COLUMNS)})
    actual_column = None
    if SOLAR_ACTUAL_COLUMN in index.columns:
        actual_column = "actual_SOLAR"
        columns[actual_column] = (index.loc[keep, SOLAR_ACTUAL_COLUMN]
                                  .to_numpy(dtype=float) * capacity_mw)
    result = pd.DataFrame(columns)
    return ForecastBundle(
        "solar", result, "MW",
        location or f"solar-zone-{location_value}", horizon, actual_column,
        metadata={
            "prediction_file": str(prediction_path),
            "index_file": str(index_path),
            "zone": location_value,
            "capacity_mw": float(capacity_mw),
            "capacity_basis": capacity_basis or (
                "ASSUMED: GEFCom2014-S publishes output as a fraction of "
                "installed capacity and never states the capacity in MW"),
            "raw_unit": "fraction of installed capacity [0, 1]",
            "actuals_attached": actual_column is not None,
        },
        physical_min=0.0, physical_max=float(capacity_mw))

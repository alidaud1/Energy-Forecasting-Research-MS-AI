"""Typed contract exchanged by the load, solar and fusion agents.

The two forecasting pipelines may use different internal state schemas, but
their hand-off to fusion must be explicit and identical.  This module contains
only validation and conversion logic; it performs no forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

QUANTILE_COLUMNS: tuple[str, ...] = tuple(f"q{i:02d}" for i in range(1, 100))


class ForecastContractError(ValueError):
    """Raised when a forecast cannot be safely consumed by fusion."""


@dataclass(frozen=True)
class ForecastBundle:
    """A validated probabilistic forecast from one domain.

    ``frame`` contains one row per timestamp and q01--q99 columns.  ``actual``
    is optional and is retained only for validation diagnostics/residual fitting.
    All timestamps are timezone-naive UTC-equivalent values and strictly
    increasing.  Values are expressed in ``unit`` for one ``location``.

    ``physical_min``/``physical_max`` record domain bounds that no realisation
    can cross (solar generation cannot be negative, nor exceed the installed
    capacity).  Fusion uses them to clamp sampled scenarios; leaving them
    ``None`` means the quantity is unbounded on that side.
    """

    source: str
    frame: pd.DataFrame
    unit: str
    location: str
    horizon: str
    actual_column: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    physical_min: float | None = None
    physical_max: float | None = None

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.unit.strip() or not self.location.strip():
            raise ForecastContractError("source, unit and location are required")
        if not self.horizon.strip():
            raise ForecastContractError("horizon is required")
        if not isinstance(self.frame, pd.DataFrame):
            raise ForecastContractError("frame must be a pandas DataFrame")
        required = {"TIMESTAMP", *QUANTILE_COLUMNS}
        missing = sorted(required.difference(self.frame.columns))
        if missing:
            raise ForecastContractError(f"missing forecast columns: {missing[:5]}")
        if self.frame.empty:
            raise ForecastContractError("forecast frame is empty")
        timestamps = pd.to_datetime(self.frame["TIMESTAMP"], errors="coerce")
        if timestamps.isna().any() or timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
            raise ForecastContractError("TIMESTAMP must be valid, unique and chronological")
        values = self.frame.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ForecastContractError("quantile forecasts contain NaN or infinite values")
        if (np.diff(values, axis=1) < -1e-10).any():
            raise ForecastContractError("quantile forecasts contain crossing quantiles")
        if self.actual_column is not None:
            if self.actual_column not in self.frame.columns:
                raise ForecastContractError(f"missing actual column: {self.actual_column}")
            actual = self.frame[self.actual_column].to_numpy(dtype=float)
            if not np.isfinite(actual).all():
                raise ForecastContractError("actual values contain NaN or infinite values")
        if self.physical_min is not None and self.physical_max is not None:
            if self.physical_min > self.physical_max:
                raise ForecastContractError("physical_min must not exceed physical_max")
        # Normalise TIMESTAMP in place on a private copy.  Validation above
        # parsed the column but earlier revisions never wrote it back, so a CSV
        # loaded bundle kept object dtype and failed to merge against a bundle
        # built from real datetimes.  Storing the parsed column makes the dtype
        # part of the contract rather than an accident of the caller's input.
        normalised = self.frame.copy()
        normalised["TIMESTAMP"] = timestamps
        object.__setattr__(self, "frame", normalised)

    @classmethod
    def from_csv(cls, path: str | Path, *, source: str, unit: str,
                 location: str, horizon: str, actual_column: str | None = None,
                 metadata: Mapping[str, Any] | None = None,
                 physical_min: float | None = None,
                 physical_max: float | None = None) -> "ForecastBundle":
        """Load and validate a canonical forecast CSV."""
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ForecastContractError(f"forecast file does not exist: {resolved}")
        return cls(source, pd.read_csv(resolved), unit, location, horizon,
                   actual_column, metadata or {}, physical_min, physical_max)

    def timestamps(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame["TIMESTAMP"])

    def quantiles(self) -> np.ndarray:
        return self.frame.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float)

    def residuals(self) -> np.ndarray | None:
        """Median-forecast residuals, or ``None`` when actuals are unavailable."""
        if self.actual_column is None:
            return None
        actual = self.frame[self.actual_column].to_numpy(dtype=float)
        return actual - self.frame["q50"].to_numpy(dtype=float)

    def replace_frame(self, frame: pd.DataFrame) -> "ForecastBundle":
        """Return a revalidated copy of this bundle carrying ``frame``."""
        return ForecastBundle(self.source, frame, self.unit, self.location,
                              self.horizon, self.actual_column, dict(self.metadata),
                              self.physical_min, self.physical_max)

    def with_prefix(self, prefix: str) -> pd.DataFrame:
        """Return timestamps and prefixed quantiles for fusion joins."""
        if not prefix.strip():
            raise ForecastContractError("prefix is required")
        result = self.frame.loc[:, ["TIMESTAMP", *QUANTILE_COLUMNS]].copy()
        result.columns = ["TIMESTAMP", *(f"{prefix}_{q}" for q in QUANTILE_COLUMNS)]
        return result

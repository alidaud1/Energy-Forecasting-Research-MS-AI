# Phase A Implementation Blueprint for a Production‑Grade Data Pipeline

## Dataset archive analysis and confirmed schemas

You uploaded **two** archives that together form your “Dataset.zip” input for Phase A:

- **Load track**: `GEFCom2014-L_V2.zip`  
- **Solar track**: `GEFCom2014-S_V2.zip`

### Confirmed zip structure

**Load zip** contains folders like:

- `Load/Task 1/L1-train.csv`, `Load/Task 1/L1-benchmark.csv`, … through Task 15  
- `Load/Solution to Task 15/...` (not needed for training)  
- `Load/Instructions.txt`

**Solar zip** contains:

- `Solar/Task 15/train15.csv`, `Solar/Task 15/predictors15.csv`, `Solar/Task 15/benchmark15.csv` (and tasks 1–14)  
- `Solar/Solution to Task 15/...` (not needed for training)  
- `Solar/Instructions.txt`

### Confirmed column names and timestamp formats

#### Load track (`Load/Task 1/L1-train.csv` and others)

Columns in `L15-train.csv` (same schema pattern across tasks):

- `ZONEID` (int)
- `TIMESTAMP` (string)
- `LOAD` (float, can be missing in early history)
- `w1 ... w25` (25 weather station temperature columns, integers)

This matches the load-track dataset description: hourly load plus **25** temperature stations. citeturn0search0

**TIMESTAMP format (load):**  
Strings like:
- `112001 1:00`
- `1112011 1:00`
- `11272011 0:00`
- `1012010 0:00`

Key properties:
- The date portion is **month+day+year concatenated without separators**, where **year is always the last 4 digits**, and month/day have **no leading zeros**.
- Hour format is `H:00` (hours can be 0–23).

This format is ambiguous for some dates (e.g., `1112011` could be 1/11/2011 or 11/1/2011), so the loader must use a **continuity-based parsing rule** (implemented below).

#### Solar track (`Solar/Task 15/train15.csv` and `predictors15.csv`)

`train15.csv` columns:
- `ZONEID`, `TIMESTAMP`, `POWER`

`predictors15.csv` columns:
- `ZONEID`, `TIMESTAMP`,
- `VAR78`, `VAR79`, `VAR134`, `VAR157`, `VAR164`, `VAR165`, `VAR166`, `VAR167`, `VAR169`, `VAR175`, `VAR178`, `VAR228`,
- plus a `POWER` column that duplicates the target (we will verify and drop it safely).

**TIMESTAMP format (solar):**  
Strings like `20120401 01:00` → **`%Y%m%d %H:%M`**, clean and deterministic.

### Why solar “decumulation” is required

GEFCom2014 solar predictors include **accumulated radiation/precipitation variables** that must be converted to hourly increments (decumulated) before modeling; published GEFCom solar solutions explicitly describe differentiating the accumulated fields during preprocessing. citeturn0search5turn0search2

A recent primary source mapping the VAR codes confirms that **VAR169, VAR175, VAR178** are radiation-energy terms (J/m²) and **VAR228** is precipitation (accumulated), which are consistent with accumulated-field behavior and resets. citeturn0search1

## Project workspace and Python environment

### Folder structure

Create this workspace layout (minimal but production-grade):

```
your-project/
  requirements.txt
  data/
    raw/
      GEFCom2014-L_V2.zip
      GEFCom2014-S_V2.zip
    processed/
      load_task1.parquet
      solar_task15.parquet
      splits/
      scalers/
  src/
    data_loaders/
      __init__.py
      io_utils.py
      time_parsing.py
      load_track_loader.py
      solar_features.py
      solar_track_loader.py
      splits.py
      scaling.py
      windowing.py
  tests/
    test_no_leakage_windows.py
```

### requirements.txt (time-series forecasting + parquet + plotting + testing)

Below is a concrete `requirements.txt` that is stable for forecasting research and supports Parquet caching. (If you run CUDA, you may need to install the torch build specific to your CUDA version.)

```txt
# Core
numpy>=1.26,<3.0
pandas>=2.1,<3.0
scipy>=1.11,<2.0

# ML / preprocessing
scikit-learn>=1.3,<2.0

# Deep learning (PyTorch)
torch>=2.2,<3.0

# Storage / IO
pyarrow>=14.0,<20.0

# Plotting
matplotlib>=3.7,<4.0

# Testing
pytest>=7.4,<9.0

# Optional but recommended (quality + progress)
tqdm>=4.66,<5.0
```

## Load and solar track loaders

Below are modular, file-by-file implementations designed to work **directly from zip** or from extracted directories later.

### src/data_loaders/io_utils.py

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import zipfile

import pandas as pd


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ZipPath:
    zip_file: Path
    internal_path: str

    def open(self):
        zf = zipfile.ZipFile(self.zip_file)
        return zf, zf.open(self.internal_path)


def read_csv_from_zip(zip_file: PathLike, internal_path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file inside a zip archive with pandas."""
    zip_file = Path(zip_file)
    with zipfile.ZipFile(zip_file) as zf:
        with zf.open(internal_path) as f:
            return pd.read_csv(f, **kwargs)


def ensure_dir(p: PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
```

### src/data_loaders/time_parsing.py

This is the key robustness piece (especially for load).

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LoadTimestampParseResult:
    timestamps: pd.DatetimeIndex
    prefer_two_digit_month_for_len3: bool
    non_hourly_steps: int


def _parse_load_timestamp_single(ts: str, prefer_two_digit_len3: bool) -> datetime:
    """
    Parse a single GEFCom load timestamp.

    Observed format examples:
      - "112001 1:00"      -> (date_part="112001",  time_part="1:00")
      - "1112011 1:00"     -> (date_part="1112011", time_part="1:00")
      - "11272011 0:00"
      - "1012010 0:00"

    date_part is month+day+year without separators; year is last 4 digits.
    For md length==3, ambiguity exists: e.g. "1112011" can be 1/11/2011 or 11/1/2011.
    """
    date_part, time_part = ts.split()
    year = int(date_part[-4:])
    md = date_part[:-4]

    hour_str, min_str = time_part.split(":")
    hour = int(hour_str)
    minute = int(min_str)

    def try_build(m: int, d: int) -> datetime | None:
        try:
            return datetime(year, m, d, hour, minute)
        except ValueError:
            return None

    # Unambiguous cases
    if len(md) == 4:
        dt = try_build(int(md[:2]), int(md[2:]))
        if dt is None:
            raise ValueError(f"Invalid md=4 parse for {ts}")
        return dt

    if len(md) == 2:
        dt = try_build(int(md[0]), int(md[1]))
        if dt is None:
            raise ValueError(f"Invalid md=2 parse for {ts}")
        return dt

    # Ambiguous case: len(md) == 3
    if len(md) != 3:
        raise ValueError(f"Unexpected load timestamp format: {ts}")

    # Candidate A: 1-digit month + 2-digit day
    cand_a = try_build(int(md[0]), int(md[1:]))
    # Candidate B: 2-digit month + 1-digit day
    cand_b = try_build(int(md[:2]), int(md[2]))

    # Choose preferred if valid; otherwise fallback
    preferred = cand_b if prefer_two_digit_len3 else cand_a
    fallback = cand_a if prefer_two_digit_len3 else cand_b

    if preferred is not None:
        return preferred
    if fallback is not None:
        return fallback

    raise ValueError(f"Both md=3 parses invalid for {ts}")


def parse_gefcom_load_timestamps(ts_series: Iterable[str]) -> LoadTimestampParseResult:
    """
    Robustly parse a full timestamp series by selecting the md=3 preference
    that yields the fewest non-hourly steps (i.e., best continuity).

    This prevents the classic ambiguity issue where early dates could parse as
    Jan 11 vs Nov 1 (example), causing huge gaps once md=4 timestamps appear.
    """
    ts_list = list(ts_series)

    def parse_with_pref(pref: bool) -> pd.DatetimeIndex:
        dts = [_parse_load_timestamp_single(ts, pref) for ts in ts_list]
        return pd.DatetimeIndex(dts)

    idx_a = parse_with_pref(pref=False)
    idx_b = parse_with_pref(pref=True)

    def non_hourly_steps(idx: pd.DatetimeIndex) -> int:
        diffs = pd.Series(idx).diff().dropna()
        # consider exactly 1 hour as expected step
        return int((diffs != pd.Timedelta(hours=1)).sum())

    n_a = non_hourly_steps(idx_a)
    n_b = non_hourly_steps(idx_b)

    if n_b < n_a:
        return LoadTimestampParseResult(timestamps=idx_b, prefer_two_digit_month_for_len3=True, non_hourly_steps=n_b)
    return LoadTimestampParseResult(timestamps=idx_a, prefer_two_digit_month_for_len3=False, non_hourly_steps=n_a)


def parse_gefcom_solar_timestamps(ts_series: Iterable[str]) -> pd.DatetimeIndex:
    """
    Solar timestamps are consistently formatted like: "20120401 01:00"
    => %Y%m%d %H:%M (deterministic).
    """
    return pd.to_datetime(list(ts_series), format="%Y%m%d %H:%M", errors="raise")
```

### src/data_loaders/load_track_loader.py

This merges load history into a continuous hourly dataframe and interpolates missing timestamps.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .io_utils import read_csv_from_zip
from .time_parsing import parse_gefcom_load_timestamps


PathLike = Union[str, Path]


@dataclass(frozen=True)
class LoadLoaderConfig:
    zip_path: Path
    task_id: int = 1  # Task 1 contains the full historical timeline in a single file
    drop_na_target: bool = True
    interpolate_missing_timestamps: bool = True
    interpolate_columns: Optional[List[str]] = None  # default: all numeric columns
    freq: str = "h"


def load_gefcom_load_task(config: LoadLoaderConfig) -> pd.DataFrame:
    """
    Returns a MultiIndex dataframe with index levels [ZONEID, timestamp].
    Columns include: LOAD, w1..w25.
    """
    task = int(config.task_id)
    internal = f"Load/Task {task}/L{task}-train.csv"

    df = read_csv_from_zip(config.zip_path, internal)

    if "ZONEID" not in df.columns or "TIMESTAMP" not in df.columns:
        raise ValueError(f"Unexpected load schema in {internal}: columns={df.columns.tolist()}")

    # Parse timestamps robustly
    parse_res = parse_gefcom_load_timestamps(df["TIMESTAMP"].astype(str).tolist())
    df = df.drop(columns=["TIMESTAMP"])
    df.insert(1, "timestamp", parse_res.timestamps)

    # Sort
    df = df.sort_values(["ZONEID", "timestamp"])

    # Optionally remove early rows where LOAD is missing (common in Task 1 historical temps)
    if config.drop_na_target and "LOAD" in df.columns:
        df = df[df["LOAD"].notnull()].copy()

    # Set MultiIndex
    df = df.set_index(["ZONEID", "timestamp"]).sort_index()

    # Reindex to continuous hourly timeline per ZONEID and interpolate gaps
    if config.interpolate_missing_timestamps:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        interp_cols = config.interpolate_columns or numeric_cols

        out = []
        for zone, g in df.groupby(level=0, sort=False):
            g = g.droplevel(0).sort_index()
            full_idx = pd.date_range(g.index.min(), g.index.max(), freq=config.freq)
            g2 = g.reindex(full_idx)

            # Interpolate selected columns on time index
            cols_to_interp = [c for c in interp_cols if c in g2.columns]
            if cols_to_interp:
                g2[cols_to_interp] = g2[cols_to_interp].interpolate(
                    method="time", limit_direction="both"
                )

            g2["ZONEID"] = zone
            g2 = g2.reset_index(names="timestamp").set_index(["ZONEID", "timestamp"])
            out.append(g2)

        df = pd.concat(out).sort_index()

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load GEFCom2014 load track and build a continuous hourly dataframe.")
    parser.add_argument("--zip_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--out_parquet", type=str, default="")
    args = parser.parse_args()

    cfg = LoadLoaderConfig(zip_path=Path(args.zip_path), task_id=args.task_id)
    df = load_gefcom_load_task(cfg)
    print(df.head())
    print(df.tail())
    print("Shape:", df.shape)

    if args.out_parquet:
        df.to_parquet(args.out_parquet)
        print("Saved:", args.out_parquet)
```

### src/data_loaders/solar_track_loader.py

This merges solar `trainXX.csv` with `predictorsXX.csv` (zone-wise alignment), parses timestamps, and outputs a MultiIndex dataframe.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .io_utils import read_csv_from_zip
from .time_parsing import parse_gefcom_solar_timestamps
from .solar_features import decumulate_solar_accumulators, SOLAR_ACCUMULATED_DEFAULT


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SolarLoaderConfig:
    zip_path: Path
    task_id: int = 15
    apply_decumulation: bool = True
    accumulated_cols: Optional[list[str]] = None
    interpolate_missing_timestamps: bool = True
    freq: str = "h"


def load_gefcom_solar_task(config: SolarLoaderConfig) -> pd.DataFrame:
    """
    Returns a MultiIndex dataframe with index levels [ZONEID, timestamp].
    Columns include POWER (target) plus VAR features.
    """
    t = int(config.task_id)
    train_path = f"Solar/Task {t}/train{t:02d}.csv" if t < 10 else f"Solar/Task {t}/train{t}.csv"
    pred_path  = f"Solar/Task {t}/predictors{t:02d}.csv" if t < 10 else f"Solar/Task {t}/predictors{t}.csv"

    train_df = read_csv_from_zip(config.zip_path, train_path)
    pred_df  = read_csv_from_zip(config.zip_path, pred_path)

    # Normalize column names
    for d in (train_df, pred_df):
        if "ZONEID" not in d.columns or "TIMESTAMP" not in d.columns:
            raise ValueError(f"Unexpected solar schema: {d.columns.tolist()}")

    # Parse timestamps deterministically
    train_ts = parse_gefcom_solar_timestamps(train_df["TIMESTAMP"].astype(str).tolist())
    pred_ts  = parse_gefcom_solar_timestamps(pred_df["TIMESTAMP"].astype(str).tolist())

    train_df = train_df.drop(columns=["TIMESTAMP"])
    pred_df  = pred_df.drop(columns=["TIMESTAMP"])
    train_df.insert(1, "timestamp", train_ts)
    pred_df.insert(1, "timestamp", pred_ts)

    # Merge on (ZONEID, timestamp)
    merged = pd.merge(
        pred_df,
        train_df[["ZONEID", "timestamp", "POWER"]],
        on=["ZONEID", "timestamp"],
        how="left",
        suffixes=("", "_y"),
        validate="many_to_one",
    )

    # If predictors file has a POWER column, validate consistency and drop it
    if "POWER" in pred_df.columns:
        if "POWER_y" in merged.columns:
            # predictors POWER is 'POWER', train POWER is 'POWER_y'
            pred_power = merged["POWER"]
            train_power = merged["POWER_y"]
            mismatch = (pred_power.notnull() & train_power.notnull() & (np.abs(pred_power - train_power) > 1e-6)).mean()
            if mismatch > 0:
                # Not fatal, but we choose train POWER as ground truth
                pass
            merged = merged.drop(columns=["POWER"])
            merged = merged.rename(columns={"POWER_y": "POWER"})
        else:
            # If merge didn't create POWER_y, keep existing POWER as target
            pass
    elif "POWER_y" in merged.columns:
        merged = merged.rename(columns={"POWER_y": "POWER"})

    # Set MultiIndex
    merged = merged.sort_values(["ZONEID", "timestamp"]).set_index(["ZONEID", "timestamp"]).sort_index()

    # Decumulate accumulated fields if requested (recommended)
    if config.apply_decumulation:
        cols = config.accumulated_cols or SOLAR_ACCUMULATED_DEFAULT
        merged = decumulate_solar_accumulators(merged, cols=cols)

    # Reindex hourly to fill missing timestamps per zone + interpolate predictors
    if config.interpolate_missing_timestamps:
        numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        # do not interpolate the target POWER by default (only optional in later stages)
        interp_cols = [c for c in numeric_cols if c != "POWER"]

        out = []
        for zone, g in merged.groupby(level=0, sort=False):
            g = g.droplevel(0).sort_index()
            full_idx = pd.date_range(g.index.min(), g.index.max(), freq=config.freq)
            g2 = g.reindex(full_idx)
            if interp_cols:
                g2[interp_cols] = g2[interp_cols].interpolate(method="time", limit_direction="both")
            g2["ZONEID"] = zone
            g2 = g2.reset_index(names="timestamp").set_index(["ZONEID", "timestamp"])
            out.append(g2)

        merged = pd.concat(out).sort_index()

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load GEFCom2014 solar track and merge train + predictors.")
    parser.add_argument("--zip_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=15)
    parser.add_argument("--out_parquet", type=str, default="")
    args = parser.parse_args()

    cfg = SolarLoaderConfig(zip_path=Path(args.zip_path), task_id=args.task_id)
    df = load_gefcom_solar_task(cfg)
    print(df.head())
    print(df.tail())
    print("Shape:", df.shape)

    if args.out_parquet:
        df.to_parquet(args.out_parquet)
        print("Saved:", args.out_parquet)
```

## Advanced preprocessing modules

### src/data_loaders/solar_features.py

Implements robust **decumulation** with reset handling.

Notes:
- GEFCom2014 solar preprocessing literature describes differentiating accumulated fields and treating precipitation specially. citeturn0search5turn0search2  
- VAR mapping from a primary/official-like reference confirms which variables are accumulated radiation/precipitation. citeturn0search1

```python
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


# Default accumulated fields for GEFCom2014 solar predictors:
# VAR169 SSRD, VAR175 STRD, VAR178 TSR, VAR228 TP (accumulated precipitation).
# (Confirmed by published GEFCom variable mapping tables.)  citeturn0search1
SOLAR_ACCUMULATED_DEFAULT: List[str] = ["VAR169", "VAR175", "VAR178", "VAR228"]


def decumulate_series_with_resets(s: pd.Series) -> pd.Series:
    """
    Convert an accumulated field into increments with reset handling.

    Rule:
      increment[t] = s[t] - s[t-1]
      if increment[t] < 0 (reset), set increment[t] = s[t]
      clip negatives to 0 to avoid small numerical artifacts

    This handles daily / model-cycle resets typical in accumulated NWP quantities.
    """
    diff = s.diff()
    # Reset handler: if diff negative, assume counter restarted -> increment equals current level
    diff = diff.where(diff >= 0, s)
    # Numerical safety
    return diff.clip(lower=0)


def decumulate_solar_accumulators(
    df: pd.DataFrame,
    cols: Iterable[str] = SOLAR_ACCUMULATED_DEFAULT,
    group_level: int = 0,
) -> pd.DataFrame:
    """
    Apply decumulation to the specified columns of a MultiIndex dataframe.
    Expected index levels: [ZONEID, timestamp].
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = out.groupby(level=group_level, sort=False)[col].apply(decumulate_series_with_resets)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional helper: add cyclical time features from the timestamp index.
    Useful for transformers (known-future covariates).
    """
    if not isinstance(df.index, pd.MultiIndex) or "timestamp" not in df.index.names:
        raise ValueError("Expected MultiIndex with level named 'timestamp'")

    ts = df.index.get_level_values("timestamp")
    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month

    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return out
```

### src/data_loaders/splits.py

Chronological splits, done **per zone** to avoid mixing time across series.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class ChronoSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split_by_zone(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
) -> ChronoSplit:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ZONEID", "timestamp"]:
        raise ValueError("Expected df indexed by MultiIndex ['ZONEID', 'timestamp']")

    trains, vals, tests = [], [], []
    for zone, g in df.groupby(level=0, sort=False):
        g = g.droplevel(0).sort_index()
        n = len(g)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_g = g.iloc[:n_train]
        val_g = g.iloc[n_train:n_train + n_val]
        test_g = g.iloc[n_train + n_val:]

        # restore ZONEID level
        for part, bucket in [(train_g, trains), (val_g, vals), (test_g, tests)]:
            part = part.copy()
            part["ZONEID"] = zone
            part = part.reset_index(names="timestamp").set_index(["ZONEID", "timestamp"])
            bucket.append(part)

    return ChronoSplit(
        train=pd.concat(trains).sort_index(),
        val=pd.concat(vals).sort_index(),
        test=pd.concat(tests).sort_index(),
    )
```

### src/data_loaders/scaling.py

Leakage-safe scaling: **fit only on train**, apply to val/test.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ScalerKind = Literal["standard", "minmax"]


@dataclass
class LeakageSafeScaler:
    feature_cols: List[str]
    target_cols: List[str]
    kind: ScalerKind = "standard"

    def __post_init__(self):
        if self.kind == "standard":
            self._x_scaler = StandardScaler()
            self._y_scaler = StandardScaler()
        elif self.kind == "minmax":
            self._x_scaler = MinMaxScaler()
            self._y_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler kind: {self.kind}")

        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "LeakageSafeScaler":
        x = train_df[self.feature_cols].to_numpy(dtype=np.float32)
        y = train_df[self.target_cols].to_numpy(dtype=np.float32)
        self._x_scaler.fit(x)
        self._y_scaler.fit(y)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit(train_df) first.")

        out = df.copy()
        out[self.feature_cols] = self._x_scaler.transform(out[self.feature_cols].to_numpy(dtype=np.float32))
        out[self.target_cols] = self._y_scaler.transform(out[self.target_cols].to_numpy(dtype=np.float32))
        return out

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit(train_df) first.")
        return self._y_scaler.inverse_transform(y_scaled)
```

## Transformer-ready windowing

### src/data_loaders/windowing.py

Generates sliding windows \((X_{hist}, X_{future}, Y)\) with configurable \(W, H, stride\).

Design choices:
- `X_hist` uses `hist_cols` over `[t-W, ..., t-1]`
- `X_future` uses `future_cols` over `[t, ..., t+H-1]`
- `Y` uses `target_cols` over `[t, ..., t+H-1]`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowConfig:
    W: int  # history length
    H: int  # horizon length
    stride: int = 1


class SlidingWindowDataset(Dataset):
    """
    Torch Dataset that returns:
      X_hist:   [W, D_hist]
      X_future: [H, D_fut]
      Y:        [H, D_y]
    Optionally returns timestamp bounds for debugging/testing.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        hist_cols: List[str],
        future_cols: List[str],
        target_cols: List[str],
        cfg: WindowConfig,
        return_times: bool = False,
    ):
        if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ZONEID", "timestamp"]:
            raise ValueError("Expected df indexed by MultiIndex ['ZONEID', 'timestamp']")
        if cfg.W <= 0 or cfg.H <= 0 or cfg.stride <= 0:
            raise ValueError("W, H, stride must be positive")

        self.df = df.sort_index()
        self.hist_cols = hist_cols
        self.future_cols = future_cols
        self.target_cols = target_cols
        self.cfg = cfg
        self.return_times = return_times

        # Precompute per-zone numpy arrays and window map
        self._zones = []
        self._zone_cache = {}  # zone -> dict of arrays/timestamps
        self._window_map: List[Tuple[int, int]] = []  # (zone, start_idx)

        for zone, g in self.df.groupby(level=0, sort=False):
            g = g.droplevel(0).sort_index()
            ts = g.index.to_numpy()
            x_hist_arr = g[self.hist_cols].to_numpy(dtype=np.float32)
            x_fut_arr = g[self.future_cols].to_numpy(dtype=np.float32)
            y_arr = g[self.target_cols].to_numpy(dtype=np.float32)

            n = len(g)
            max_start = n - (cfg.W + cfg.H)
            if max_start < 0:
                continue

            self._zones.append(zone)
            self._zone_cache[zone] = {
                "ts": ts,
                "x_hist": x_hist_arr,
                "x_fut": x_fut_arr,
                "y": y_arr,
            }
            for start in range(0, max_start + 1, cfg.stride):
                self._window_map.append((zone, start))

        if len(self._window_map) == 0:
            raise ValueError("No windows available. Check W/H vs data length.")

    def __len__(self) -> int:
        return len(self._window_map)

    def window_time_bounds(self, idx: int):
        zone, start = self._window_map[idx]
        cache = self._zone_cache[zone]
        ts = cache["ts"]
        W, H = self.cfg.W, self.cfg.H
        hist_start = pd.Timestamp(ts[start])
        hist_end = pd.Timestamp(ts[start + W - 1])
        fut_start = pd.Timestamp(ts[start + W])
        fut_end = pd.Timestamp(ts[start + W + H - 1])
        return zone, hist_start, hist_end, fut_start, fut_end

    def __getitem__(self, idx: int):
        zone, start = self._window_map[idx]
        cache = self._zone_cache[zone]
        W, H = self.cfg.W, self.cfg.H

        X_hist = cache["x_hist"][start:start + W]                   # [W, D_hist]
        X_future = cache["x_fut"][start + W:start + W + H]          # [H, D_fut]
        Y = cache["y"][start + W:start + W + H]                     # [H, D_y]

        X_hist_t = torch.from_numpy(X_hist)
        X_future_t = torch.from_numpy(X_future)
        Y_t = torch.from_numpy(Y)

        if self.return_times:
            _, hs, he, fs, fe = self.window_time_bounds(idx)
            return X_hist_t, X_future_t, Y_t, (hs, he, fs, fe)

        return X_hist_t, X_future_t, Y_t
```

## Validation, splits, and a unit test for leakage

### tests/test_no_leakage_windows.py

This test verifies:
- Train/val/test are chronological per zone
- Windowing produces **strictly non-overlapping** history vs future time bounds (no leakage)

```python
import pandas as pd
import numpy as np

from src.data_loaders.splits import chronological_split_by_zone
from src.data_loaders.windowing import SlidingWindowDataset, WindowConfig


def test_chronological_split_and_no_leakage():
    # Build a small synthetic multi-zone dataset
    idx = pd.date_range("2023-01-01", periods=200, freq="h")
    df_zone1 = pd.DataFrame(
        {
            "feat": np.arange(len(idx), dtype=np.float32),
            "target": np.arange(len(idx), dtype=np.float32) * 2.0,
        },
        index=idx,
    )
    df_zone1["ZONEID"] = 1
    df_zone1 = df_zone1.reset_index(names="timestamp").set_index(["ZONEID", "timestamp"])

    df_zone2 = df_zone1.copy()
    df_zone2 = df_zone2.reset_index()
    df_zone2["ZONEID"] = 2
    df_zone2 = df_zone2.set_index(["ZONEID", "timestamp"])

    df = pd.concat([df_zone1, df_zone2]).sort_index()

    # Split
    split = chronological_split_by_zone(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

    # Check chronological ordering per zone
    for zone, g_tr in split.train.groupby(level=0):
        tr_max = g_tr.droplevel(0).index.max()
        va_min = split.val.xs(zone, level=0).index.min()
        te_min = split.test.xs(zone, level=0).index.min()
        assert tr_max < va_min
        assert va_min < te_min

    # Windowing on train set
    cfg = WindowConfig(W=24, H=12, stride=3)
    ds = SlidingWindowDataset(
        split.train,
        hist_cols=["feat", "target"],
        future_cols=["feat"],      # example known future feature
        target_cols=["target"],
        cfg=cfg,
        return_times=True,
    )

    # Check multiple random windows
    for i in [0, len(ds)//3, len(ds)-1]:
        _, _, _, (hs, he, fs, fe) = ds[i]
        assert he < fs, "History end must be strictly before future start (no leakage)"
        assert (fs - he) == pd.Timedelta(hours=1), "Future must start exactly 1 hour after hist end"
        assert fe >= fs
```

## What to do next after you paste this into your repo

You can now run Phase A end-to-end like this:

1) Put the two zips under `data/raw/`:
- `data/raw/GEFCom2014-L_V2.zip`
- `data/raw/GEFCom2014-S_V2.zip`

2) Generate parquet caches:

```bash
python -m src.data_loaders.load_track_loader \
  --zip_path data/raw/GEFCom2014-L_V2.zip \
  --task_id 1 \
  --out_parquet data/processed/load_task1.parquet

python -m src.data_loaders.solar_track_loader \
  --zip_path data/raw/GEFCom2014-S_V2.zip \
  --task_id 15 \
  --out_parquet data/processed/solar_task15.parquet
```

3) Run unit tests:

```bash
pytest -q
```

## Notes on scope alignment with your thesis goals

- This Phase A pipeline is explicitly designed around GEFCom2014’s **hourly load + 25 weather stations** citeturn0search0 and solar’s **NWP predictors + accumulated fields that require decumulation** citeturn0search5turn0search2turn0search1.
- The output MultiIndex format is intentionally compatible with later “agentic” orchestration: your future agents can operate at the level of *config → artifact* (parquet + scalers + window datasets) without re-implementing technical details.

If you want, the next step is **Phase A.2**: a single `prepare_data.py` pipeline runner that loads → decumulates → adds time features → splits → scales → writes a `torch` tensor cache (or memory-mapped arrays) and a `dataset_manifest.json` so your agents can work from a stable contract.
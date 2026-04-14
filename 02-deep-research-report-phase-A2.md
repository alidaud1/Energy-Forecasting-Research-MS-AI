# Phase A Master Orchestrator for GEFCom2014 Load + Solar Data Preparation

## Executive overview

This final Phase‑A “glue” script will turn your two raw archives (**GEFCom2014 Load** and **GEFCom2014 Solar**) into reproducible, model‑ready artifacts in **one command**:

- Loads raw CSVs (from zip) → merges into continuous hourly **MultiIndex** DataFrames (`ZONEID`, `timestamp`)
- Applies **time feature engineering** (sin/cos encodings)
- Solar: applies **decumulation** to accumulated meteorological variables (critical in GEFCom solar preprocessing) citeturn1search0
- Performs chronological **train/val/test** splits per zone
- Fits scalers **only on Train** (leakage-safe), then transforms Val/Test
- Persists:
  - `.parquet` files (clean + split + scaled)
  - scaler objects (`joblib`)
  - a `dataset_manifest.json` containing column lists, scaler stats (mean/std), and split date ranges

Two key design references:
- The GEFCom2014 **load** track includes hourly load (2005–2010) and temperature data starting 2001 from **25 stations**, with hourly resolution. citeturn0search1  
- The GEFCom2014 **solar** track preprocessing in published solutions explicitly uses **decumulation** of accumulated variables (and special handling for precipitation). citeturn1search0  

## Confirmed dataset schemas and date formats from your zip files

This orchestration code assumes (and has been designed around) the **actual** schemas found inside your uploaded zips:

### Load track (`GEFCom2014-L_V2.zip`)
- File pattern: `Load/Task {k}/L{k}-train.csv`
- Columns:
  - `ZONEID`, `TIMESTAMP`, `LOAD`, `w1..w25` (25 weather station temperature columns)
- Timestamp format (**nonstandard, ambiguous**):
  - Strings like `112001 1:00`, `1112011 1:00`, `11272011 0:00`
  - The date part is `month + day + year` with **no separators** and variable digit lengths (6/7/8).  
  - The loader’s `parse_gefcom_load_timestamps()` resolves ambiguity by choosing the parsing mode that yields the most continuous hourly sequence.

### Solar track (`GEFCom2014-S_V2.zip`)
- Files: `Solar/Task {k}/train{k}.csv` and `Solar/Task {k}/predictors{k}.csv`
- `train15.csv` columns: `ZONEID`, `TIMESTAMP`, `POWER`
- `predictors15.csv` columns:
  - `ZONEID`, `TIMESTAMP`, **VAR** predictors (`VAR78`, `VAR79`, `VAR134`, …, `VAR228`), plus a duplicate `POWER` column
- Timestamp format (clean, deterministic):
  - Strings like `20120401 01:00` → `%Y%m%d %H:%M`

## Orchestrator responsibilities and artifacts

### What `src/prepare_data.py` will produce

It writes the following structure:

```
data/processed/
  load/
    clean.parquet
    train.parquet
    val.parquet
    test.parquet
    train_scaled.parquet
    val_scaled.parquet
    test_scaled.parquet
  solar/
    clean.parquet
    train.parquet
    val.parquet
    test.parquet
    train_scaled.parquet
    val_scaled.parquet
    test_scaled.parquet
  scalers/
    load_scaler.joblib
    solar_scaler.joblib
  dataset_manifest.json
```

### Why YAML config + `safe_load` and why joblib

- For YAML configs, `yaml.safe_load()` is recommended because `yaml.load()` can construct arbitrary Python objects and can be unsafe for untrusted YAML input. citeturn0search0  
- For scaler persistence:
  - scikit‑learn’s official “model persistence” docs show `joblib.dump()` / `joblib.load()` as a supported persistence method (with security warnings similar to pickle). citeturn0search3turn0search2  
  - joblib’s persistence docs explain that it’s efficient for objects holding large NumPy arrays, but it is still pickle-based and must not load untrusted files. citeturn0search4  

## Deliverable code for `src/prepare_data.py`

Below is a complete, production‑grade script designed to work with the Phase‑A modules you implemented earlier (`load_track_loader`, `solar_track_loader`, `solar_features.add_time_features`, `splits`, `scaling`).

> Assumptions about your codebase (matches your Phase‑A modules):
> - `src/data_loaders/load_track_loader.py` exposes `LoadLoaderConfig`, `load_gefcom_load_task`
> - `src/data_loaders/solar_track_loader.py` exposes `SolarLoaderConfig`, `load_gefcom_solar_task`
> - `src/data_loaders/solar_features.py` exposes `add_time_features`
> - `src/data_loaders/splits.py` exposes `chronological_split_by_zone`
> - `src/data_loaders/scaling.py` exposes `LeakageSafeScaler`

```python
# src/prepare_data.py
from __future__ import annotations

import argparse
import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import yaml

# Phase-A modules (you implemented earlier)
from src.data_loaders.load_track_loader import LoadLoaderConfig, load_gefcom_load_task
from src.data_loaders.solar_track_loader import SolarLoaderConfig, load_gefcom_solar_task
from src.data_loaders.solar_features import add_time_features
from src.data_loaders.splits import chronological_split_by_zone
from src.data_loaders.scaling import LeakageSafeScaler


LOGGER = logging.getLogger("prepare_data")


TIME_FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]


@dataclass
class PrepConfig:
    # Inputs
    load_zip: str = "data/raw/GEFCom2014-L_V2.zip"
    solar_zip: str = "data/raw/GEFCom2014-S_V2.zip"
    load_task_id: int = 1
    solar_task_id: int = 15

    # Preprocessing
    add_time_features: bool = True
    solar_decumulate: bool = True

    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20

    # Scaling
    scaler_kind: str = "standard"  # {"standard","minmax"}

    # Feature policy (important)
    load_future_include_weather: bool = False  # realistic default: future time features only

    # Outputs
    out_dir: str = "data/processed"

    # Manifest
    manifest_name: str = "dataset_manifest.json"


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        # safe_load is recommended by PyYAML docs for safety
        return yaml.safe_load(f) or {}


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _split_range(df: pd.DataFrame) -> Dict[str, str]:
    ts = df.index.get_level_values("timestamp")
    return {
        "start": pd.Timestamp(ts.min()).isoformat(),
        "end": pd.Timestamp(ts.max()).isoformat(),
    }


def _split_range_by_zone(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for zone, g in df.groupby(level=0, sort=False):
        g = g.droplevel(0)
        out[str(zone)] = {
            "start": pd.Timestamp(g.index.min()).isoformat(),
            "end": pd.Timestamp(g.index.max()).isoformat(),
        }
    return out


def _infer_cols_load(df: pd.DataFrame, include_weather_future: bool) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns: feature_cols, target_cols, hist_cols, future_cols
    """
    target_cols = ["LOAD"]

    weather_cols = [c for c in df.columns if c.startswith("w")]
    # Features exclude target to avoid double scaling
    feature_cols = weather_cols + ([c for c in TIME_FEATURE_COLS if c in df.columns])

    # For windowing later:
    hist_cols = ["LOAD"] + weather_cols + ([c for c in TIME_FEATURE_COLS if c in df.columns])

    # Default: only time features are known-in-advance.
    if include_weather_future:
        future_cols = weather_cols + ([c for c in TIME_FEATURE_COLS if c in df.columns])
    else:
        future_cols = [c for c in TIME_FEATURE_COLS if c in df.columns]

    return feature_cols, target_cols, hist_cols, future_cols


def _infer_cols_solar(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns: feature_cols, target_cols, hist_cols, future_cols
    """
    target_cols = ["POWER"]
    var_cols = [c for c in df.columns if c.startswith("VAR")]
    time_cols = [c for c in TIME_FEATURE_COLS if c in df.columns]

    # Features exclude target to avoid double scaling
    feature_cols = var_cols + time_cols

    hist_cols = ["POWER"] + var_cols + time_cols
    future_cols = var_cols + time_cols
    return feature_cols, target_cols, hist_cols, future_cols


def _extract_scaler_stats(scaler_wrapper: LeakageSafeScaler) -> Dict[str, Any]:
    """
    Returns a JSON-serializable dict with per-column mean/std (StandardScaler)
    or min/max (MinMaxScaler), for both features and targets.
    """
    def stats_for(prefix: str, cols: List[str], sk_scaler: Any) -> Dict[str, Any]:
        # StandardScaler
        if hasattr(sk_scaler, "mean_") and hasattr(sk_scaler, "scale_"):
            return {
                "type": "StandardScaler",
                "columns": cols,
                "mean": {c: float(m) for c, m in zip(cols, sk_scaler.mean_)},
                "std": {c: float(s) for c, s in zip(cols, sk_scaler.scale_)},
            }
        # MinMaxScaler
        if hasattr(sk_scaler, "data_min_") and hasattr(sk_scaler, "data_max_"):
            return {
                "type": "MinMaxScaler",
                "columns": cols,
                "data_min": {c: float(mn) for c, mn in zip(cols, sk_scaler.data_min_)},
                "data_max": {c: float(mx) for c, mx in zip(cols, sk_scaler.data_max_)},
            }
        # Fallback
        return {"type": type(sk_scaler).__name__, "columns": cols}

    # LeakageSafeScaler stores these attributes by our Phase-A design:
    #   _x_scaler and _y_scaler
    x_scaler = getattr(scaler_wrapper, "_x_scaler", None)
    y_scaler = getattr(scaler_wrapper, "_y_scaler", None)

    return {
        "kind": scaler_wrapper.kind,
        "features": stats_for("features", scaler_wrapper.feature_cols, x_scaler),
        "targets": stats_for("targets", scaler_wrapper.target_cols, y_scaler),
    }


def _save_parquets(track_dir: Path, df_clean: pd.DataFrame, splits_raw: Dict[str, pd.DataFrame], splits_scaled: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    _ensure_dir(track_dir)

    clean_path = track_dir / "clean.parquet"
    df_clean.to_parquet(clean_path)
    paths["clean"] = str(clean_path.as_posix())

    for k, v in splits_raw.items():
        p = track_dir / f"{k}.parquet"
        v.to_parquet(p)
        paths[k] = str(p.as_posix())

    for k, v in splits_scaled.items():
        p = track_dir / f"{k}_scaled.parquet"
        v.to_parquet(p)
        paths[f"{k}_scaled"] = str(p.as_posix())

    return paths


def _prepare_one_track(
    name: str,
    df_clean: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    hist_cols: List[str],
    future_cols: List[str],
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    scaler_kind: str,
) -> Dict[str, Any]:
    # Splits
    split = chronological_split_by_zone(
        df_clean,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    splits_raw = {"train": split.train, "val": split.val, "test": split.test}

    # Fit scaler on TRAIN ONLY (leakage-safe)
    scaler = LeakageSafeScaler(
        feature_cols=feature_cols,
        target_cols=target_cols,
        kind=scaler_kind,
    ).fit(split.train)

    splits_scaled = {
        "train": scaler.transform(split.train),
        "val": scaler.transform(split.val),
        "test": scaler.transform(split.test),
    }

    # Persist parquet
    track_dir = out_dir / name
    parquet_paths = _save_parquets(track_dir, df_clean, splits_raw, splits_scaled)

    # Persist scaler
    scalers_dir = _ensure_dir(out_dir / "scalers")
    scaler_path = scalers_dir / f"{name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Manifest entry
    manifest = {
        "name": name,
        "paths": {
            **parquet_paths,
            "scaler": str(scaler_path.as_posix()),
        },
        "columns": {
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "hist_cols": hist_cols,
            "future_cols": future_cols,
        },
        "splits": {
            "global_ranges": {
                "train": _split_range(split.train),
                "val": _split_range(split.val),
                "test": _split_range(split.test),
            },
            "per_zone_ranges": {
                "train": _split_range_by_zone(split.train),
                "val": _split_range_by_zone(split.val),
                "test": _split_range_by_zone(split.test),
            },
        },
        "scaler_stats": _extract_scaler_stats(scaler),
        "dataset_stats": {
            "n_rows_clean": int(len(df_clean)),
            "n_rows_train": int(len(split.train)),
            "n_rows_val": int(len(split.val)),
            "n_rows_test": int(len(split.test)),
            "n_zones": int(df_clean.index.get_level_values("ZONEID").nunique()),
            "columns_clean": list(df_clean.columns),
        },
    }
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-A data preparation: Load + Solar tracks -> parquet + scalers + dataset_manifest.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="", help="YAML config path (optional).")

    # Paths
    p.add_argument("--load_zip", type=str, default=None)
    p.add_argument("--solar_zip", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    # Tasks
    p.add_argument("--load_task_id", type=int, default=None)
    p.add_argument("--solar_task_id", type=int, default=None)

    # Behavior flags
    g1 = p.add_mutually_exclusive_group()
    g1.add_argument("--add_time_features", dest="add_time_features", action="store_true")
    g1.add_argument("--no_add_time_features", dest="add_time_features", action="store_false")
    p.set_defaults(add_time_features=None)

    g2 = p.add_mutually_exclusive_group()
    g2.add_argument("--solar_decumulate", dest="solar_decumulate", action="store_true")
    g2.add_argument("--no_solar_decumulate", dest="solar_decumulate", action="store_false")
    p.set_defaults(solar_decumulate=None)

    g3 = p.add_mutually_exclusive_group()
    g3.add_argument("--load_future_include_weather", dest="load_future_include_weather", action="store_true")
    g3.add_argument("--load_future_time_only", dest="load_future_include_weather", action="store_false")
    p.set_defaults(load_future_include_weather=None)

    # Splits + scaling
    p.add_argument("--train_ratio", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--scaler_kind", type=str, default=None, choices=["standard", "minmax"])

    return p


def _finalize_config(args: argparse.Namespace) -> PrepConfig:
    base = PrepConfig().__dict__
    yaml_cfg = _load_yaml_config(args.config)
    merged = _deep_update(base, yaml_cfg)

    # Override with CLI args if provided
    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is not None:
            merged[key] = val

    return PrepConfig(**merged)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_arg_parser().parse_args()
    cfg = _finalize_config(args)

    out_dir = _ensure_dir(Path(cfg.out_dir))

    LOGGER.info("Starting Phase-A prepare_data with config: %s", cfg)

    # -------------------------
    # LOAD TRACK
    # -------------------------
    load_df = load_gefcom_load_task(
        LoadLoaderConfig(zip_path=Path(cfg.load_zip), task_id=cfg.load_task_id)
    )

    if cfg.add_time_features:
        load_df = add_time_features(load_df)

    load_feature_cols, load_target_cols, load_hist_cols, load_future_cols = _infer_cols_load(
        load_df, include_weather_future=cfg.load_future_include_weather
    )

    # -------------------------
    # SOLAR TRACK
    # -------------------------
    solar_df = load_gefcom_solar_task(
        SolarLoaderConfig(
            zip_path=Path(cfg.solar_zip),
            task_id=cfg.solar_task_id,
            apply_decumulation=cfg.solar_decumulate,
        )
    )

    if cfg.add_time_features:
        solar_df = add_time_features(solar_df)

    solar_feature_cols, solar_target_cols, solar_hist_cols, solar_future_cols = _infer_cols_solar(solar_df)

    # -------------------------
    # Persist + manifest per track
    # -------------------------
    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "name": "phase_a_prepare_data",
            "version": "1.0.0",
        },
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "inputs": {
            "load_zip": cfg.load_zip,
            "solar_zip": cfg.solar_zip,
            "load_task_id": cfg.load_task_id,
            "solar_task_id": cfg.solar_task_id,
        },
        "preprocessing": {
            "add_time_features": cfg.add_time_features,
            "solar_decumulate": cfg.solar_decumulate,
            "load_future_include_weather": cfg.load_future_include_weather,
            "time_feature_cols": TIME_FEATURE_COLS,
        },
        "splits": {
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "test_ratio": cfg.test_ratio,
        },
        "scaling": {
            "scaler_kind": cfg.scaler_kind,
        },
        "tracks": {},
    }

    manifest["tracks"]["load"] = _prepare_one_track(
        name="load",
        df_clean=load_df,
        feature_cols=load_feature_cols,
        target_cols=load_target_cols,
        hist_cols=load_hist_cols,
        future_cols=load_future_cols,
        out_dir=out_dir,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        scaler_kind=cfg.scaler_kind,
    )

    manifest["tracks"]["solar"] = _prepare_one_track(
        name="solar",
        df_clean=solar_df,
        feature_cols=solar_feature_cols,
        target_cols=solar_target_cols,
        hist_cols=solar_hist_cols,
        future_cols=solar_future_cols,
        out_dir=out_dir,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        scaler_kind=cfg.scaler_kind,
    )

    # -------------------------
    # Write dataset_manifest.json
    # -------------------------
    manifest_path = out_dir / cfg.manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info("Phase A complete: parquet + scalers + manifest are ready in %s", out_dir)


if __name__ == "__main__":
    main()
```

## Sample `dataset_manifest.json` structure

This is a representative structure of what the script writes (values shown as examples / placeholders):

```json
{
  "created_at_utc": "2026-04-12T09:15:22.123456+00:00",
  "pipeline": { "name": "phase_a_prepare_data", "version": "1.0.0" },
  "system": { "python": "3.11.8", "platform": "Linux-6.x-x86_64" },
  "inputs": {
    "load_zip": "data/raw/GEFCom2014-L_V2.zip",
    "solar_zip": "data/raw/GEFCom2014-S_V2.zip",
    "load_task_id": 1,
    "solar_task_id": 15
  },
  "preprocessing": {
    "add_time_features": true,
    "solar_decumulate": true,
    "load_future_include_weather": false,
    "time_feature_cols": ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]
  },
  "splits": { "train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2 },
  "scaling": { "scaler_kind": "standard" },
  "tracks": {
    "load": {
      "name": "load",
      "paths": {
        "clean": "data/processed/load/clean.parquet",
        "train": "data/processed/load/train.parquet",
        "val": "data/processed/load/val.parquet",
        "test": "data/processed/load/test.parquet",
        "train_scaled": "data/processed/load/train_scaled.parquet",
        "val_scaled": "data/processed/load/val_scaled.parquet",
        "test_scaled": "data/processed/load/test_scaled.parquet",
        "scaler": "data/processed/scalers/load_scaler.joblib"
      },
      "columns": {
        "feature_cols": ["w1","w2","...","w25","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"],
        "target_cols": ["LOAD"],
        "hist_cols": ["LOAD","w1","...","w25","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"],
        "future_cols": ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]
      },
      "splits": {
        "global_ranges": {
          "train": {"start": "2005-01-01T01:00:00", "end": "2009-..."},
          "val": {"start": "2009-...", "end": "2010-..."},
          "test": {"start": "2010-...", "end": "2010-10-01T00:00:00"}
        },
        "per_zone_ranges": { "train": { "1": {"start":"...","end":"..."} } }
      },
      "scaler_stats": {
        "kind": "standard",
        "features": {
          "type": "StandardScaler",
          "columns": ["w1","w2","..."],
          "mean": {"w1": 55.1, "w2": 54.7},
          "std": {"w1": 12.3, "w2": 11.8}
        },
        "targets": {
          "type": "StandardScaler",
          "columns": ["LOAD"],
          "mean": {"LOAD": 100.2},
          "std": {"LOAD": 25.4}
        }
      },
      "dataset_stats": { "n_rows_clean": 123456, "n_zones": 1, "columns_clean": ["LOAD","w1","..."] }
    },
    "solar": {
      "name": "solar",
      "...": "same structure with POWER + VAR columns"
    }
  }
}
```

## README section for one-command execution

Copy/paste this into your `README.md` under **Phase A**:

```md
## Phase A — Data Preparation (one command)

This repository includes a Phase‑A pipeline that converts the raw GEFCom2014 zips into:
- cleaned parquet datasets (clean/train/val/test and scaled variants),
- persisted scalers (joblib),
- a `dataset_manifest.json` describing columns, split ranges, and scaler statistics.

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If using YAML configs, ensure PyYAML is installed:
```bash
pip install pyyaml
```

### Run with a YAML config
```bash
python src/prepare_data.py --config config/default_prep.yaml
```

### Run with flags only
```bash
python src/prepare_data.py \
  --load_zip data/raw/GEFCom2014-L_V2.zip \
  --solar_zip data/raw/GEFCom2014-S_V2.zip \
  --out_dir data/processed \
  --load_task_id 1 \
  --solar_task_id 15 \
  --solar_decumulate \
  --add_time_features \
  --scaler_kind standard
```

Artifacts will be written to:
- `data/processed/load/`
- `data/processed/solar/`
- `data/processed/scalers/`
- `data/processed/dataset_manifest.json`
```

### Optional: example `config/default_prep.yaml`

If you want an immediate config file to use, create:

```yaml
load_zip: data/raw/GEFCom2014-L_V2.zip
solar_zip: data/raw/GEFCom2014-S_V2.zip
out_dir: data/processed

load_task_id: 1
solar_task_id: 15

add_time_features: true
solar_decumulate: true

train_ratio: 0.70
val_ratio: 0.10
test_ratio: 0.20

scaler_kind: standard

# realistic default for load: only time features are known in the future
load_future_include_weather: false
```

This completes Phase A. When you’re ready, Phase B can consume the manifest to auto‑select columns and scalers without hardcoding anything.
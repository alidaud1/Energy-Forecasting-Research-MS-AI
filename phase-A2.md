# Phase A2 (One-Command Orchestrator) - Flow, Inputs, Outputs, Verification

This document describes Phase A2 as implemented in this repo. It is the “glue” layer that runs Phase A1-style loading + preprocessing + splitting + scaling and persists reproducible artifacts in one command.

Code entrypoint:
- `src/prepare_data.py`

Dependencies (installed via `requirements.txt`):
- `pandas`, `pyarrow` (parquet IO)
- `scikit-learn`, `joblib` (scalers + persistence)
- `PyYAML` (optional YAML config)

## Goal

Run one command to produce:
- Clean, model-ready parquet datasets for both tracks (Load + Solar)
- Chronological train/val/test splits (per zone)
- Leakage-safe scaling (fit on train only, apply to val/test)
- Persisted scaler objects (`joblib`)
- A `dataset_manifest.json` that records paths, column policies, split ranges, and scaler stats

## Inputs

Phase A2 can load from either:
- `.zip` archives, or
- extracted dataset root folders

Configured by:
- CLI flags to `python -m src.prepare_data ...`, and/or
- a YAML config via `--config config/default_prep.yaml`

### Required paths

Load track root must contain:
- `Load/Task {k}/L{k}-train.csv`

Solar track root must contain:
- `Solar/Task {k}/train{k}.csv`
- `Solar/Task {k}/predictors{k}.csv`

Default config in this repo points to extracted folders:
- `../dataset/GEFCom2014-L_V2`
- `../dataset/GEFCom2014-S_V2`

## Outputs (Desired Artifact Layout)

Phase A2 writes to `out_dir` (default: `data/processed/`):

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

## Code Flow (What `src/prepare_data.py` Does)

### 1) Read + Clean (per track)

Load track:
- Calls `load_gefcom_load_task(LoadLoaderConfig(...))` from `src/data_loaders/load_track_loader.py`.
- Produces a MultiIndex DataFrame indexed by `['ZONEID','timestamp']`.

Solar track:
- Calls `load_gefcom_solar_task(SolarLoaderConfig(...))` from `src/data_loaders/solar_track_loader.py`.
- Produces a MultiIndex DataFrame indexed by `['ZONEID','timestamp']`.

Optional time features:
- If enabled, calls `add_time_features(df)` from `src/data_loaders/solar_features.py` for both tracks.
- Adds `hour_sin,hour_cos,dow_sin,dow_cos,month_sin,month_cos`.

Solar decumulation:
- Controlled by `solar_decumulate` in config/flags.
- Uses the solar loader’s decumulation path to convert accumulated VAR fields into hourly increments.

### 2) Infer Column Policies (per track)

Load:
- `feature_cols`: `w1..w25` + time features
- `target_cols`: `LOAD`
- `hist_cols`: `LOAD` + `w1..w25` + time features
- `future_cols`:
  - default is time-only (realistic; future weather not known)
  - if `load_future_include_weather=true`, includes `w1..w25` as future covariates

Solar:
- `feature_cols`: all `VAR*` + time features
- `target_cols`: `POWER`
- `hist_cols`: `POWER` + `VAR*` + time features
- `future_cols`: all `VAR*` + time features (predictors exist into the forecast horizon)

These lists are written into the manifest and are intended to be the contract for later windowing/training steps.

### 3) Split Chronologically (per zone)

- Uses `chronological_split_by_zone(df, train_ratio, val_ratio, test_ratio)` from `src/data_loaders/splits.py`.
- Splits are strictly time-ordered inside each `ZONEID`.

### 4) Scale Leakage-Safely

- Builds `LeakageSafeScaler(feature_cols, target_cols, kind=...)` from `src/data_loaders/scaling.py`.
- Fits only on the train split.
- Transforms train/val/test.

Artifacts written:
- `train_scaled.parquet`, `val_scaled.parquet`, `test_scaled.parquet`
- scalers saved under `data/processed/scalers/*.joblib`

### 5) Persist + Manifest

Per track:
- Writes `clean.parquet` plus split parquets (raw + scaled).
- Writes the scaler object as joblib.

Global:
- Writes `data/processed/dataset_manifest.json` containing:
  - all artifact paths
  - column lists (feature/target/hist/future)
  - split date ranges (global + per-zone)
  - scaler stats (mean/std or min/max depending on scaler kind)
  - dataset stats (row counts, zone counts, columns)

## Verified Status (Your Run)

You ran:
- `python -m src.prepare_data --config config/default_prep.yaml`

And validated key outputs:

Load clean:
- `data/processed/load/clean.parquet` shape `(50376, 32)`
- index `['ZONEID','timestamp']`
- `na_LOAD = 0`

Solar clean:
- `data/processed/solar/clean.parquet` shape `(59112, 19)`
- index `['ZONEID','timestamp']`
- `na_POWER = 2160`
  - expected: this corresponds to 1 forecast month across 3 zones (`720 * 3`)

Manifest:
- `data/processed/dataset_manifest.json` exists and includes `paths`, column policies, split ranges, and scaler stats for both tracks.

## How To Run (Windows PowerShell)

From `D:\Tests\02\coding_implementation\Phase-A-Implementation`:

Install/update dependencies:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run with config:

```powershell
python -m src.prepare_data --config config/default_prep.yaml
```

Run with flags only:

```powershell
python -m src.prepare_data `
  --load_zip "..\dataset\GEFCom2014-L_V2" `
  --solar_zip "..\dataset\GEFCom2014-S_V2" `
  --out_dir "data/processed" `
  --load_task_id 1 `
  --solar_task_id 15 `
  --solar_decumulate `
  --add_time_features `
  --scaler_kind standard `
  --load_future_time_only
```

## Quick Output Checks

```powershell
python -c "import json; m=json.load(open('data/processed/dataset_manifest.json')); print(m['inputs']); print(m['tracks'].keys())"
python -c "import pandas as pd; df=pd.read_parquet('data/processed/load/clean.parquet'); print(df.shape, df.index.names, df['LOAD'].isna().sum())"
python -c "import pandas as pd; df=pd.read_parquet('data/processed/solar/clean.parquet'); print(df.shape, df.index.names, df['POWER'].isna().sum())"
```


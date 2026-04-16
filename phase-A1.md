# Phase A1 (Data Pipeline) - Flow, Inputs, Outputs

This document describes what Phase A1 does in this repo, what files it reads, what it produces, and the execution flow through the code.

## Goal

Convert raw GEFCom2014 V2 CSVs into stable, model-ready artifacts:
- Time-indexed (hourly) tables with consistent schema
- Solar accumulated predictors decumulated into hourly increments
- Ready for Phase A2/B steps: chronological splits, leakage-safe scaling, sliding-window datasets

## Code Location

Phase A1 implementation lives under:
- `src/data_loaders/`
- `tests/`

Key modules:
- `src/data_loaders/load_track_loader.py`
- `src/data_loaders/solar_track_loader.py`
- `src/data_loaders/time_parsing.py`
- `src/data_loaders/solar_features.py`
- `src/data_loaders/splits.py`
- `src/data_loaders/scaling.py`
- `src/data_loaders/windowing.py`

## Inputs

The loaders accept `--zip_path` as either:
1. A `.zip` file path, or
2. An extracted dataset root folder path.

### Expected internal paths (relative to zip root or extracted root)

Load:
- `Load/Task {k}/L{k}-train.csv`

Solar:
- `Solar/Task {k}/train{k}.csv`
- `Solar/Task {k}/predictors{k}.csv`

Typical local dataset folder (this repo):
- `../dataset/GEFCom2014-L_V2`
- `../dataset/GEFCom2014-S_V2`

## Outputs

Phase A1 produces parquet cache(s) under:
- `data/processed/`

### Load artifact

`data/processed/load_task1.parquet`
- Index: MultiIndex `['ZONEID', 'timestamp']`
- Columns: `LOAD` + `w1..w25` (26 columns)
- Properties:
  - hourly index per zone
  - `LOAD` contains no missing values after `drop_na_target=True`

### Solar artifact

`data/processed/solar_task15.parquet`
- Index: MultiIndex `['ZONEID', 'timestamp']`
- Columns: `POWER` + `VAR78..VAR228` (13 columns total: 1 target + 12 predictors)
- Properties:
  - zones: 1..3
  - predictors exist through the forecast month, so `POWER` is missing for that month
  - accumulated variables are decumulated by zone:
    - `VAR169`, `VAR175`, `VAR178`, `VAR228`

## Execution Flow (What Happens When You Run the Commands)

### A) Load flow (`src/data_loaders/load_track_loader.py`)

1. Read `Load/Task {task_id}/L{task_id}-train.csv` from zip or extracted root (`src/data_loaders/io_utils.py`).
2. Parse `TIMESTAMP` into a real datetime index (`src/data_loaders/time_parsing.py`):
   - Load timestamps include ambiguous month/day encodings.
   - Parsing uses continuity optimization to preserve an hourly sequence.
3. Sort by `ZONEID` and timestamp.
4. If `drop_na_target=True`, drop rows where `LOAD` is missing (Task 1 has an early block without `LOAD`).
5. Set MultiIndex `['ZONEID', 'timestamp']`.
6. If `interpolate_missing_timestamps=True`:
   - for each zone, reindex to a continuous hourly `date_range`
   - interpolate numeric columns on time
7. Write parquet if `--out_parquet` is supplied.

### B) Solar flow (`src/data_loaders/solar_track_loader.py`)

1. Read `Solar/Task {task_id}/train{task_id}.csv` and `Solar/Task {task_id}/predictors{task_id}.csv`.
2. Parse deterministic solar timestamps (`%Y%m%d %H:%M`).
3. Merge predictors with target on `(ZONEID, timestamp)`:
   - predictors cover both training and the future benchmark month
   - target `POWER` is missing for the forecast month (expected)
4. Decumulate accumulated predictors per zone (`src/data_loaders/solar_features.py`):
   - compute hourly increments
   - handle resets (negative diffs) by using the current value
   - clip negative artifacts to 0
5. If `interpolate_missing_timestamps=True`:
   - reindex to continuous hourly per zone
   - interpolate predictors only (not `POWER`)
6. Write parquet if `--out_parquet` is supplied.

## How To Run (Windows PowerShell)

From `D:\Tests\02\coding_implementation\Phase-A-Implementation`:

1) Install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

2) Generate Load cache (using extracted dataset root):

```powershell
python -m src.data_loaders.load_track_loader `
  --zip_path "D:\Tests\02\coding_implementation\dataset\GEFCom2014-L_V2" `
  --task_id 1 `
  --out_parquet "data/processed/load_task1.parquet"
```

3) Generate Solar cache (using extracted dataset root):

```powershell
python -m src.data_loaders.solar_track_loader `
  --zip_path "D:\Tests\02\coding_implementation\dataset\GEFCom2014-S_V2" `
  --task_id 15 `
  --out_parquet "data/processed/solar_task15.parquet"
```

4) Run tests:

```powershell
pytest -q
```

## Quick Sanity Checks

```powershell
python -c "import pandas as pd; df=pd.read_parquet('data/processed/load_task1.parquet'); print(df.index.names, df.shape, df.columns[:6])"
python -c "import pandas as pd; df=pd.read_parquet('data/processed/solar_task15.parquet'); print(df.index.names, df.shape, df.columns[:6])"
```


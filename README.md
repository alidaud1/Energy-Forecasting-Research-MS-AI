# Energy-Forecasting-Research-MS-AI
A Multi-Agent Transformer-Based Framework for Joint Load and Solar Energy Forecasting Using Agentic AI

## Probabilistic fusion and decision-support module

The `fusion/` package validates and aligns independently generated load and
solar forecast artifacts, creates net-load scenarios, and produces deterministic
decision indicators. It includes a synchronized synthetic benchmark and frozen
example outputs for reproducing the fusion evaluation. GEFCom load and solar
competition tracks are not coincident measurements; results from cyclically
aligned tracks must not be described as measured joint-grid forecasts.

Start with [`fusion/README.md`](fusion/README.md) for setup, benchmark commands,
input forecast format, assumptions, and test instructions. Generated runs and
local API keys are excluded by `.gitignore`; the optional key template is
`fusion/.env.example`.

## Phase A (Data Pipeline) - A1

This directory implements the Phase A1 blueprint from `Planning/01-deep-research-report-phase-A1.md`:
- Load + Solar dataset loaders
- Solar decumulation utilities
- Chronological splits per zone
- Leakage-safe scaling
- Transformer-ready sliding windows
- Unit test that checks split ordering + no-leakage window bounds

### Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Data location

You can point loaders at either:
- the original `.zip` archives, or
- an extracted dataset folder root that contains `Load/...` or `Solar/...`.

Expected internal paths:
- Load: `Load/Task {k}/L{k}-train.csv`
- Solar: `Solar/Task {k}/train{k}.csv` and `Solar/Task {k}/predictors{k}.csv`

### Example: build parquet caches

Using zip archives in `data/raw/`:

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

Using extracted folders (example paths):

```bash
python -m src.data_loaders.load_track_loader \
  --zip_path ../dataset/GEFCom2014-L_V2 \
  --task_id 1 \
  --out_parquet data/processed/load_task1.parquet

python -m src.data_loaders.solar_track_loader \
  --zip_path ../dataset/GEFCom2014-S_V2 \
  --task_id 15 \
  --out_parquet data/processed/solar_task15.parquet
```

### Tests

```bash
pytest -q
```

## Phase A (Data Pipeline) - A2 (One-Command Orchestrator)

Phase A2 adds a single orchestrator that runs:
- load + solar ingestion
- optional time features
- solar decumulation
- per-zone chronological split (train/val/test)
- leakage-safe scaling (fit on train only)
- persistence of parquet + scalers + `dataset_manifest.json`

### Run (recommended)

```bash
python -m src.prepare_data --config config/default_prep.yaml
```

### Run (flags only)

```bash
python -m src.prepare_data \
  --load_zip ../dataset/GEFCom2014-L_V2 \
  --solar_zip ../dataset/GEFCom2014-S_V2 \
  --out_dir data/processed \
  --load_task_id 1 \
  --solar_task_id 15 \
  --solar_decumulate \
  --add_time_features \
  --scaler_kind standard
```

Artifacts written:
- `data/processed/load/` (`clean.parquet`, `train.parquet`, `val.parquet`, `test.parquet`, and `_scaled` variants)
- `data/processed/solar/` (same set)
- `data/processed/scalers/*.joblib`
- `data/processed/dataset_manifest.json`

## Phase B (Agentic Baselines) - B2

Phase B2 adds a runnable multi-agent baseline loop (prepare -> train -> evaluate) driven by the MessagePool.

Install Phase B extras:

```bash
pip install -r requirements_phase_b.txt
```

Run:

```bash
python orchestrator.py --track load --model xgboost
```

Outputs:
- `logs/message_pool.jsonl`
- `logs/runs/<run_id>/models/*.joblib`
- `logs/runs/<run_id>/plots/*.png`

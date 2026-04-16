# Energy-Forecasting-Research-MS-AI
A Multi-Agent Transformer-Based Framework for Joint Load and Solar Energy Forecasting Using Agentic AI

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

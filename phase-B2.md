# Phase B2 - Agentic Baseline Orchestration (Prepare -> Train -> Evaluate)

Phase B2 builds a minimal, runnable multi-agent loop on top of Phase B1 (Message schema + MessagePool) and Phase A2 artifacts (parquets + scalers + manifest).

The baseline loop:

`Task configuration -> Prepare -> Ready -> Model plan -> Train -> Evaluate`

## Inputs

Required Phase A2 artifacts:
- `data/processed/dataset_manifest.json`
- For selected `track` (`load` or `solar`):
  - `train_scaled.parquet`, `val_scaled.parquet`, `test_scaled.parquet`
  - `scaler.joblib`

Runtime parameters (passed to orchestrator):
- `--track` (`load` or `solar`)
- `--model` (`linear` or `xgboost`)
- window config: `--W`, `--H`, `--stride`
- limits: `--max_train_windows`, `--max_test_windows`
- optional Optuna skeleton: `--hpo --n_trials`

Dependencies for Phase B2:
- install base: `pip install -r requirements.txt`
- install Phase B extras: `pip install -r requirements_phase_b.txt`

## Outputs

Audit log:
- `logs/message_pool.jsonl` (append-only JSONL of all messages)

Run artifacts (per run):
- `logs/runs/<run_id>/models/<track>_<model>.joblib`
- `logs/runs/<run_id>/plots/<track>_<model>_pred_vs_actual.png`

Final evaluation is published as a message:
- topic: `evaluate/<run_id>`
- kind: `result`
- payload: `metrics` (MAE/RMSE) and artifact paths

## Components and Responsibilities

### MessagePool (Phase B1)
- In-memory pub/sub routing by topic type + run scope
- JSONL persistence for auditability

### Agents (Phase B2)

`src/agents/task_manager_agent.py`
- Publishes:
  - `system/<run_id>` event `run_started`
  - `prepare/<run_id>` command `prepare` including `run_config` (model + window + training settings)

`src/agents/preparation_agent.py`
- Reads `dataset_manifest.json`
- Validates required artifacts exist and are readable
- Resolves relative paths robustly (relative to manifest directory)
- Publishes:
  - `prepare/<run_id>` event `ready` including `paths`, `columns`, and the forwarded `run_config`

`src/agents/model_manager_agent.py`
- Converts `ready` into a training plan
- Chooses baseline hyperparams:
  - `linear`: `fit_intercept=True`
  - `xgboost`: default tree params (hist method)
- Optional Optuna ask/tell skeleton:
  - samples a trial and injects sampled params into the `train` command
  - on `evaluate` result, calls `study.tell(trial, rmse)`
- Publishes:
  - `train/<run_id>` command `train`

`src/agents/model_developer_agent.py`
- Loads `train_scaled.parquet` and `test_scaled.parquet`
- Builds tabular baseline training data from time windows:
  - Uses `SlidingWindowDataset` (`src/data_loaders/windowing.py`)
  - Flattens window tensors into a 2D `X` per window
  - Flattens multi-horizon targets into a wide `Y`
- Trains:
  - `linear`: `LinearRegression`
  - `xgboost`: `XGBRegressor` wrapped with `MultiOutputRegressor`
- Evaluates on test:
  - MAE and RMSE over all horizons (flattened)
  - Saves a plot for horizon step 1
- Publishes:
  - `evaluate/<run_id>` `result` `evaluation_complete` (metrics + paths)

## Important Correctness Details

Solar target missingness:
- Solar `POWER` is missing for the forecast month in the clean data.
- When building windows, the ModelDeveloper skips any window with non-finite (NaN/inf) targets.

Inverse-scaling for multi-horizon targets:
- Targets are scaled per-timepoint in the dataframe (single target column).
- Windowing flattens `H * D_y` targets into a wide vector.
- The ModelDeveloper reshapes to `(-1, n_targets)` before inverse-transform and then reshapes back, so metrics/plots are in the original units.

## How to Run (Windows PowerShell)

From `D:\Tests\02\coding_implementation\Phase-A-Implementation`:

1) Install dependencies:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements_phase_b.txt
```

2) Run orchestrator (example: load + xgboost):

```powershell
python orchestrator.py --track load --model xgboost --W 168 --H 24 --stride 24
```

3) Run orchestrator (example: solar + linear):

```powershell
python orchestrator.py --track solar --model linear --W 168 --H 24 --stride 24
```

Expected message sequence in `logs/message_pool.jsonl`:
- `system/<run_id>`: `run_started` (event)
- `prepare/<run_id>`: `prepare` (command)
- `prepare/<run_id>`: `ready` (event)
- `train/<run_id>`: `train` (command)
- `evaluate/<run_id>`: `evaluation_complete` (result) or an `error`


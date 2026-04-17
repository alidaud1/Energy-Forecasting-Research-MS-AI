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

## What Happens When You Run XGBoost (End-to-End Flow)

Command example:

```powershell
python orchestrator.py --track load --model xgboost --print_messages
```

### Step 0: Orchestrator Setup

1. `orchestrator.py` generates a new `run_id` (example: `run_20260416_173849`).
2. It creates `MessagePool(log_path="logs/message_pool.jsonl", load_existing=True)`.
   - Because `load_existing=True`, the pool loads prior JSONL history into memory; if you print messages, you may see messages from earlier runs too.
3. It instantiates 4 agents (all scoped to the same `run_id`):
   - `TaskManagerAgent` (role: `task_manager`)
   - `PreparationAgent` (role: `preparation`)
   - `ModelManagerAgent` (role: `model_manager`)
   - `ModelDeveloperAgent` (role: `model_developer`)
4. It enters a heartbeat loop and calls `tick()` on each agent repeatedly.

### Step 1: TaskManager Starts the Run

On its first tick only, TaskManager publishes two messages:

1. `system/<run_id>` event: `run_started`
   - Includes: track, model_type, window config, training limits, and hpo config
2. `prepare/<run_id>` command: `prepare`
   - Includes: `manifest_path`, `track`, and the same `run_config`

### Step 2: PreparationAgent Validates Phase-A Artifacts

PreparationAgent receives the `prepare` command and:

1. Opens the `dataset_manifest.json`.
2. Validates the selected track exists (`load` or `solar`).
3. Resolves the required artifact paths and checks they exist and are readable:
   - `train_scaled.parquet`
   - `val_scaled.parquet`
   - `test_scaled.parquet`
   - `scaler.joblib`
4. Publishes:
   - `prepare/<run_id>` event: `ready` (includes resolved `paths`, `columns`, and forwarded `run_config`)
   - or `prepare/<run_id>` error: `not_ready` (with an error string)

### Step 3: ModelManager Builds a Training Plan

ModelManager receives the `ready` event and publishes:

1. `train/<run_id>` command: `train`
   - Includes:
     - `paths` (full resolved artifact paths)
     - `columns` (feature/target/hist/future col lists)
     - `model_type` (here: `xgboost`)
     - baseline hyperparams (defaults in `src/agents/model_manager_agent.py`)
     - window config (`W`, `H`, `stride`)
     - training limits (`max_train_windows`, `max_test_windows`)
     - optional Optuna info (ask/tell skeleton) if `--hpo` enabled

### Step 4: ModelDeveloper Trains and Evaluates the Baseline

ModelDeveloper receives the `train` command and does:

#### 4.1 Load scaled datasets and scaler
- Loads `train_scaled.parquet` and `test_scaled.parquet`.
- Loads the scaler object from `scaler.joblib` (for inverse-transforming predictions/targets into original units for metrics/plots).

#### 4.2 Build sliding windows, then flatten to tabular X/Y
Uses `SlidingWindowDataset` from `src/data_loaders/windowing.py`:

- For each window:
  - `X_hist` shape: `[W, D_hist]` from `hist_cols`
  - `X_future` shape: `[H, D_future]` from `future_cols`
  - `Y` shape: `[H, D_y]` from `target_cols`
- Convert to a single tabular row:
  - `X = flatten(X_hist) + flatten(X_future)` (concatenated)
  - `y = flatten(Y)`

For **load** with Phase A2 defaults:
- `D_hist = len(hist_cols) = 32` (`LOAD` + `w1..w25` + 6 time features)
- `D_future = len(future_cols) = 6` (time features only, because `load_future_include_weather=false`)
- `D_y = len(target_cols) = 1`

So the feature vector length is:
- `X_dim = W*D_hist + H*D_future`
  - with defaults `W=168`, `H=24`: `X_dim = 168*32 + 24*6 = 5520`
And the target vector length is:
- `y_dim = H*D_y = 24`

The agent publishes `train/<run_id>` events while building windows (`window_build_progress`) and around fitting (`model_fit_started`, `model_fit_complete`).

#### 4.3 How many models are trained for XGBoost?

We use:
- `xgboost.XGBRegressor` as the base estimator
- wrapped by `sklearn.multioutput.MultiOutputRegressor`

`MultiOutputRegressor` trains **one separate model per output dimension**.

In this baseline:
- output dimension = `y_dim = H * D_y`
- for load, `D_y=1`, so `y_dim = H`

So with default `H=24`, it trains **24 separate XGBoost models** (one per horizon step).
That is why it can feel slow: you are effectively fitting 24 boosted tree models, each with a high-dimensional feature vector.

#### 4.4 Predict, inverse-scale, metric, and plot

1. Predict: `y_pred = model.predict(X_test)` gives shape `[n_windows, y_dim]`.
2. Inverse-scale:
   - targets were scaled per-timepoint (1 target column), but windowing flattened them into `y_dim` columns,
   - so we reshape to `(-1, n_targets)` before inverse-transform and reshape back.
3. Metrics:
   - MAE and RMSE computed on flattened arrays (all horizons).
4. Plot:
   - saves a `Predicted vs Actual` line plot for horizon step 1 (column 0).
5. Persist artifacts:
   - model: `logs/runs/<run_id>/models/<track>_xgboost.joblib`
   - plot: `logs/runs/<run_id>/plots/<track>_xgboost_pred_vs_actual.png`
6. Publish evaluation:
   - `evaluate/<run_id>` result: `evaluation_complete` with `metrics` + artifact paths

### Step 5: Orchestrator Completes

The orchestrator watches `evaluate/<run_id>`:
- If it sees `evaluation_complete` (result), it prints `[OK] Run completed`.
- If it sees any `error` message on `prepare/train/evaluate`, it prints the error payload and stops.
- If training is too slow, it can hit `--timeout_seconds` and print a timeout warning.


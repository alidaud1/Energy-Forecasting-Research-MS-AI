# Beginner Guide: Joint Load + Solar Forecasting Workbench

This project is a team of small “helpers” (agents) that pass notes to each other in a shared notebook. Each helper does one job, in order, to turn energy data into a forecast you can review and adjust.

The research title is: **“A Multi-Agent Transformer-Based Framework for Joint Load and Solar Energy Forecasting Using Agentic AI.”**

## 1) Step-by-Step System Flow (from data to final forecast)

### Step 0: What you start with (inputs)
There are two kinds of inputs:

- **Prepared datasets** (recommended): the system reads `data/processed/dataset_manifest.json` and the Parquet files it points to (train/val/test splits).
- **Raw GEFCom CSVs**: these are handled earlier by the Phase-A preparation script (`python -m src.prepare_data ...`) which creates the “prepared datasets” above.

In short: **Raw CSVs → Phase A creates Parquet + manifest → Agents use the manifest.**

### Step 1: Preparation Phase (PreparationAgent)
Think of this helper as the “data inspector and feature builder”.

What it does:

- Opens the manifest file and checks all required files exist.
- Loads the prepared train/val/test data.
- Prints a short summary (row counts, time range, missing target values).
- Optionally builds extra “math features” based on **Table III** if you enable `feature_search`.

What “math features” means (simple examples):

- **Trigonometric time features**: instead of writing “hour=23”, it writes smooth circles like `hour_sin` and `hour_cos`. This helps models learn daily cycles without “hard edges” at midnight.
- **Lag features**: copies of a value from earlier hours (example: “what was the load 24 hours ago?”).

Important: the Table III feature search is **optional**.

- If you do not enable it, the agent uses the features already present in Phase-A prepared files.
- If you enable it, the agent creates **run-specific** prepared data under:
  `logs/runs/<run_id>/prepared/<track>/...`
  so it does not overwrite your main dataset.

### Step 2: Optimization Phase (ModelManagerAgent)
Think of this helper as the “brain that tries different recipes”.

It uses Optuna to try combinations from **Table II** (model settings), like:

- XGBoost: number of trees, tree depth, learning rate
- Linear: no regularization vs ridge, and ridge strength
- SVR: C and gamma
- MLP/LSTM/GRU/CNN: size and learning rate settings

It does this in **batches**:

- It “asks” Optuna for one or more trial recipes.
- It tells the training helper to train and score each one.
- After results come back, it “tells” Optuna the scores so the next recipes improve.

So the loop is basically: **try recipe → measure → learn → try a better recipe**.

### Step 3: Training Phase (ModelDeveloperAgent)
Think of this helper as the “builder”.

It receives a training request and:

- Builds the model (XGBoost, SVR, Linear, or a small neural network like MLP/LSTM/GRU/CNN).
- Trains it on `train_scaled`.
- Checks accuracy:
  - If optimization (HPO) is enabled, it scores on `val_scaled` (so it doesn’t “cheat” using test data).
  - Otherwise, it typically reports test scoring for the chosen plan.
- Writes artifacts (saved model, plots, predictions) into the run folder:
  `logs/runs/<run_id>/...`

### Step 4: Integration & Deployment (ForecastIntegrationAgent + DeploymentOperatorAgent)
Think of these helpers as “final assembly + your adjustments”.

ForecastIntegrationAgent:

- Collects results for load and solar.
- Writes the final report:
  `logs/runs/<run_id>/reports/joint_decision_report.md`
- If load and solar overlap in time, it can compute **net-load** (load minus solar). If they do not overlap, it correctly reports “not computed”.

DeploymentOperatorAgent:

- Applies your post-processing rules (Table I style) to the predictions, and saves an “adjusted” forecast.
- Keeps both versions:
  - `y_pred_original`
  - `y_pred` (after adjustment)

Examples of adjustments you can request:

- Simple scaling: “multiply predictions by a factor”
- Simple bias: “add a fixed number”
- Time-only scaling: “only adjust certain hours”
- Threshold trigger: “only adjust if predicted load is above a threshold”
- External trigger: “only adjust if some variable is above a threshold”

## 2) Interactive Front-End Guide (React Workbench)

The React Workbench is a simple control room. It shows:

- Which run you are looking at
- The live messages between helpers
- Charts and reports
- A chat box to send commands

### Header buttons

- **Workbench**: the main dashboard with charts, report, and chat.
- **Agent Flow**: a live “conversation diagram” showing the helpers passing messages.
- **Refresh**: reloads the run list.
- **Start Run**: starts a new research run using default settings defined in the frontend.

### Runs panel (left side)

- **Select run_id** dropdown: choose which run’s messages and artifacts to display.
- **status** text: shows the run status.

### Workbench tab: View Controls

- **variant**
  - `best available`: show whatever the system thinks is best for the run
  - `proposed`: the model’s original forecast
  - `human_adjusted`: your adjusted forecast (after post-processing)
- **horizon (1..24)**: choose which future hour you are viewing (1 = next hour, 24 = 24 hours ahead).
- **Overlay variants**: show proposed and human_adjusted together (two lines).
- **Overlay baseline**: show a baseline model line together with the main model line.
- **Refresh charts**: fetches predictions again from the backend and redraws the charts.

### Forecast charts (Load and Solar)
These charts draw:

- **Actual** (real observed value)
- **Predicted** (the model’s forecast)
- Optional overlay lines (another variant or baseline) if enabled

### Report panel
Shows the markdown report created by the system, usually:
`logs/runs/<run_id>/reports/joint_decision_report.md`

### Visualization Panel (explains the “why”)

- **Net-Load**
  - If computed: shows a plot and metrics
  - If not computed: tells you the reason (usually “no overlap in time”)
- **Loss Curve (Train vs Val)**
  - Shows how transformer training improved (lower is better)
- **LLM Token Usage**
  - Only appears if you run with LLM enabled; otherwise it stays empty
- **Hyperparameter Importance (Optuna)**
  - Only appears if you ran baseline optimization with at least 2 trials (`--hpo --n_trials >= 2`)
  - It answers: “which settings mattered most?”
- **Load Error vs Hour-of-Day**
  - A scatter plot that helps you see if the model is worse at certain hours

### Agent Flow page (Sequence Diagram)
This is a live “message map”.

- Each vertical column is a helper (a “lifeline”).
- Each arrow is a note being passed (a command, event, result, or error).
- **Reset Cursor** button:
  - Think of this as “start reading messages from the beginning again”.
  - Useful if you want to replay a run’s conversation from scratch.

### Gate popups (Human-in-the-loop)
Sometimes the system pauses and asks you to approve.

You will see a modal called **Human Gate** with:

- A **prompt** describing what’s about to happen
- Buttons:
  - **Approve**: continue
  - **Send**: send the text you typed (or “approve” if empty)

### Chat / Feedback (what you can type)
You can type short “commands”. Common examples:

- Approve/reject:
  - `approve`
  - `reject`
- Change training plan:
  - `set epochs=5 batch_size=128 lr=0.001 patience=2`
  - `run load only`
  - `run solar only`
  - `run both`
- Post-processing adjustments (Table I style):
  - `accept`
  - `scale load factor=1.05`
  - `bias solar add=-0.1`
  - `scale load hours=18-21 factor=1.05`
  - `scale load factor=1.05 theta=5000`
    - Meaning: “Only if the predicted load is at least 5000, increase it by 5%.”
  - `scale solar factor=0.95 phi_col=VAR169 phi_theta=0.7`
    - Meaning: “Only if VAR169 is at least 0.7, reduce solar forecast by 5%.”

Note: if you want to adjust for something like “tomorrow is cloudy”, you can express it as a rule:

- “Cloudy day → reduce solar forecast”
  - Example: `scale solar factor=0.85 hours=10-16`
  - Or use an external variable trigger if you have a cloud-related column available as `phi_col`.

## 3) Real-World Examples (simple stories)

### Example A: Load Forecasting (heatwave)
Story: “Tomorrow is unusually hot, people will use more air-conditioning.”

How you run it (conceptually):

1. Use a baseline model like XGBoost (fast and strong).
2. Add lag-like features so the model sees recent patterns.
3. If you have a temperature column in your dataset, you can enable temperature lag features and interactions in feature_search.

If the forecast still looks low during evening peak hours, you can apply a targeted adjustment:

- `scale load hours=18-21 factor=1.05`

### Example B: Solar Forecasting (sunny vs cloudy)
Story: “Clouds suddenly reduce solar output.”

How the system helps:

- A CNN/LSTM/GRU baseline can learn patterns in recent history windows.
- If your data includes a helpful external signal (cloud cover proxy, irradiance, etc.), you can use it as a trigger:
  - `scale solar factor=0.85 phi_col=<your_cloud_signal> phi_theta=<threshold>`

If you have no external signal, you can still do a simple time-window adjustment:

- `scale solar hours=10-16 factor=0.9`

## One simple mental model

- **PreparationAgent**: checks ingredients and (optionally) cuts/chops them into better shapes (features).
- **ModelManagerAgent**: picks recipes to try (optimization).
- **ModelDeveloperAgent**: cooks the recipe (training) and tastes it (accuracy).
- **Integration + Deployment**: plates the final dish, and lets you add salt/pepper (scaling/bias rules) in a traceable way.


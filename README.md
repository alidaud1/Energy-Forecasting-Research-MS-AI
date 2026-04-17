# A Multi-Agent Transformer-Based Framework for Joint Load and Solar Energy Forecasting Using Agentic AI

This repository is a research prototype that uses a **team of software agents** to prepare energy datasets, train forecasting models (including transformers), and generate decision-ready reports for **electric load** and **solar power** forecasting.

It is designed to be understandable as a “pipeline that runs itself”:
- One part prepares the data correctly.
- Another part trains forecasting models.
- Another part combines results into a single report.
- All steps leave an auditable trace in logs so results are reproducible.

## What Problem This System Solves (Plain Language)

Electric grids need forecasts of:
- **Load**: how much electricity people will demand.
- **Solar generation**: how much electricity solar panels will produce.

If both are available for the same location and same time, you can also compute:
- **Net-load = Load − Solar**

This system forecasts load and solar, and tries to compute net-load when it is scientifically valid (same timestamps and compatible zones).

## What Data It Uses

The system is built around the **GEFCom2014** competition datasets:
- A load track dataset (hourly load + weather-like variables)
- A solar track dataset (hourly solar power + predictor variables)

Important note:
- The GEFCom load and solar tracks are not guaranteed to overlap in time or represent the same “grid system”.
- Because of that, **net-load may be impossible to compute** for many runs. The system will state this clearly in the final report.

## What “Agentic AI” Means Here

Instead of one big script doing everything, the system uses multiple small “agents” that communicate through messages.

Think of agents like roles in a project team:
- A manager assigns tasks.
- A data checker validates inputs.
- A model developer trains models.
- An integrator writes the final report.

Each message is logged, so you can later explain “what happened and why”.

## High-Level End-to-End Flow

1. **Prepare data** (cleaning, time features, splitting into train/validation/test, scaling).
2. **Run forecasting** for load and solar:
   - Baseline model (XGBoost) optionally
   - Transformer model (PatchTST-like or iTransformer-like)
3. **Integrate results** into a single set of reports and plots:
   - Best per-track results
   - Baseline vs transformer comparison plots
   - Net-load if alignment exists
   - A system audit appendix from the message log

## The Agents (What Each One Does)

### 1) TaskManagerAgent
Starts a run and asks for data preparation.
- Output: “prepare” command messages.

### 2) PreparationAgent
Checks that prepared datasets exist and match expectations (correct files, columns, split order).
- Output: “ready” messages that include where the prepared datasets live.

### 3) LoadForecastingAgent
Specializes the plan for load forecasting and converts a high-level “train” plan into a concrete “train_execute” request for transformer training.
- Output: “train_execute” command for the model developer.

### 4) SolarForecastingAgent
Same role as load forecasting, but with solar-specific checks and settings.
- Output: “train_execute” command for the model developer.

### 5) ModelDeveloperAgent
This is the “trainer”.
It can do two kinds of training:
- **Baseline training** (XGBoost/Linear): traditional tabular model.
- **Transformer training** (PatchTST-like / iTransformer-like): sequence model.

It writes the core artifacts for each model:
- trained model file
- prediction plot (predicted vs actual)
- a predictions Parquet file used later for alignment and analysis

### 6) ForecastIntegrationAgent
This is the “research report writer”.
It waits until it has enough results (load + solar, at least one transformer each), then produces:
- a joint report
- a final research summary
- plots comparing baseline vs transformer errors
- net-load metrics and plots (only if alignment exists)
- a system audit appendix

## What Files Get Produced (Outputs You Should Look At)

All run outputs are stored under:
- `logs/runs/<run_id>/`

Typical outputs include:
- `models/`
  - Transformer checkpoints (`*.pt`)
  - Baseline model files (`*.joblib`)
- `predictions/`
  - One Parquet per model that contains a time-indexed forecast table (used for net-load and comparisons)
- `plots/`
  - Predicted vs actual plots
  - Baseline-vs-transformer error distribution plots (when baselines were run)
  - Triple-axis timeline plot (only when net-load is computable)
- `reports/`
  - `joint_decision_report.md`
  - `final_research_summary.md`
  - `system_performance_audit.md`

## How To Run (Simple)

From the project folder with your Python environment active:

1) Confirm the prepared datasets are valid:
```powershell
python scripts/phase_c_precheck.py --manifest data/processed/dataset_manifest.json
```

2) Run the research pipeline (transformers for load + solar):
```powershell
python orchestrator.py --mode research --print_messages --timeout_seconds 7200
```

3) Recommended for Phase C3 reporting (adds baselines so plots can compare):
```powershell
python orchestrator.py --mode research --with_baselines --print_messages --timeout_seconds 7200
```

## How To Interpret “Net-Load Not Computed”

Net-load is only meaningful if:
- load and solar refer to the same place/system, and
- the predictions cover the same timestamps (and zones, if zones exist).

If there is no overlap, the system does **not** force net-load (that would be misleading research).
Instead, it reports:
- load forecast quality (MAE/RMSE)
- solar forecast quality (MAE/RMSE)
- a clear statement that net-load was not computed due to non-overlap

## What “Success” Looks Like

A run is considered successful when:
- data precheck passes,
- the run produces model artifacts under `logs/runs/<run_id>/`,
- the integration agent writes `reports/final_research_summary.md`,
- and there are no error messages for that run in `logs/message_pool.jsonl`.

## One-Sentence Summary

This system is a message-driven, multi-agent research pipeline that prepares energy datasets, trains both baseline and transformer forecasters for load and solar, and generates an auditable thesis-ready report that attempts net-load validation only when scientifically valid.

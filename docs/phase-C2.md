# Phase C2 - Transformer Training + Specialized Forecasting Agents (Research Mode)

Phase C2 upgrades the system from tabular baselines (Phase B2) to true sequence models that consume 3D tensors:
- `X` shaped `[batch, sequence_length, channels]`
- `Y` shaped `[batch, horizon, 1]`

It adds:
- A minimal PatchTST-like forecaster (patch tokens)
- A minimal iTransformer-like forecaster (variates as tokens)
- A transformer execution path (`train_execute`) that trains on GPU if available
- Specialized forecasting agents for load and solar
- An integration agent that writes a joint report and optionally net-load if there is overlap

## New/Updated Files

Models:
- `src/models/transformers.py`

Agents:
- `src/agents/load_forecasting_agent.py`
- `src/agents/solar_forecasting_agent.py`
- `src/agents/forecast_integration_agent.py`
- `src/agents/model_developer_agent.py` (extended: baseline `train` + transformer `train_execute`)

Orchestrator:
- `orchestrator.py` now supports `--mode baseline` and `--mode research`

## Message Flow (Research Mode)

1. Two task managers publish `prepare` commands (load + solar).
2. `PreparationAgent` validates Phase A2 artifacts and publishes `ready` for each track.
3. Orchestrator injects two `train` commands (load + solar) with transformer settings.
4. `LoadForecastingAgent` and `SolarForecastingAgent` translate `train` -> directed `train_execute`.
5. `ModelDeveloperAgent` trains the transformer, writes artifacts, and publishes `evaluate` results.
6. `ForecastIntegrationAgent` waits for both results, optionally computes net-load, writes a joint report, and publishes `integrate/joint_report_ready`.

## How to Run

Baseline mode (unchanged):

```powershell
python orchestrator.py --mode baseline --track load --model xgboost --print_messages
```

Research mode (transformers, parallel load + solar):

```powershell
python orchestrator.py --mode research --manifest data/processed/dataset_manifest.json --print_messages --timeout_seconds 7200
```

By default, the orchestrator runs a **Phase C precheck** (verifies Phase A2 parquet invariants like hourly continuity, split order, and NA policy) before starting. To skip it:

```powershell
python orchestrator.py --mode research --no_precheck
```

Model selection:
- `--load_model patchtst|itransformer`
- `--solar_model patchtst|itransformer`

Training parameters:
- `--epochs`, `--batch_size`, `--lr`, `--patience`

Artifacts:
- `logs/runs/<run_id>/models/*.pt`
- `logs/runs/<run_id>/plots/*.png`
- `logs/runs/<run_id>/predictions/*.parquet`
- `logs/runs/<run_id>/reports/joint_decision_report.md`

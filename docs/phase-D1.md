# Phase D1 - Human-in-the-Loop (HITL) Integration

Phase D1 adds the core thesis highlight: **interactive control** over a multi-agent forecasting run.
It does not change the underlying forecasting math; it adds **gates** where a human can approve or edit what the agents do next.

## What’s Implemented

Interactive mode is available in **research runs**:

```powershell
python orchestrator.py --mode research --interactive --with_baselines --timeout_seconds 7200
```

The run now follows this order:
1. **Preparation stage** runs on the real Phase A2 artifacts.
2. **Prep Gate** appears once both tracks are ready.
3. **Model Gate** appears after Prep Gate approval.
4. Training runs exactly as in Phase C3 (real baselines + real transformers).
5. Integration produces the same Phase C3 outputs and reports.
6. If the plan was edited at the gates, training artifacts are written under a **child run folder** and linked to the parent run.

## Gates

### 1) Prep Gate (`prep_gate_v1`)
Trigger: when `PreparationAgent` has published `ready` for all selected tracks.

What you see:
- row counts, zone counts, timestamp ranges for `train_scaled`, `val_scaled`, `test_scaled`
- target missingness count (e.g., solar forecast month may contain NaNs in test)

Allowed commands:
- `approve` / `reject`
- `run load only` / `run solar only` / `run both`
- `enable baselines` / `disable baselines`
- `set W=168 H=24 stride=24`

### 2) Model Gate (`model_gate_v1`)
Trigger: after Prep Gate approval.

What you see:
- chosen tracks, windowing config, baseline on/off
- transformer model selections and training hyperparameters

Allowed commands:
- `approve` / `reject`
- `set epochs=5 batch_size=128 lr=0.001 patience=2`
- `set load_model=itransformer solar_model=patchtst`
- `enable baselines` / `disable baselines`
- `run load only` / `run solar only` / `run both`

### 3) Post-Process Gate (`postprocess_gate_v1`)
Trigger: after forecasts and the Phase C3 integration artifacts are available.

What you see:
- best load/solar model identity
- whether net-load could be computed

Allowed commands:
- `accept`
- `scale load factor=1.05`
- `scale solar factor=0.95`
- `scale load hours=18-21 factor=1.05`
- `bias load add=10`
- `bias solar add=-0.1`

What happens:
- The adjustment is applied to the *best model’s* prediction parquet (`y_pred` only).
- Outputs are written under: `logs/runs/<run_id>/child_runs/<child_id>/...`
- Originals are not overwritten.

## How The HITL Messaging Works

- Agents request gates by publishing:
  - `topic=gate/<run_id>` and payload `{event: gate_requested, gate_id, stage, prompt, plan}`
- The orchestrator prints the prompt and waits for terminal input, then publishes:
  - `topic=user_feedback/<run_id>` and payload `{event: user_feedback, gate_id, stage, human_text}`
- The interactive task manager applies edits, re-prompts if needed, and only proceeds once approved.

## Files Added / Modified

Added:
- `src/agents/interactive_task_manager_agent.py`

Modified:
- `orchestrator.py` (adds `--interactive` and the CLI pause/resume loop)
- `src/agents/preparation_agent.py` (adds split summaries to `ready` messages for the Prep Gate)
- `src/agents/forecast_integration_agent.py` (adds Post-Process Gate and child-run adjusted artifacts)

## Next Steps (Phase D1 remaining)

None. Phase D1 is complete (Prep Gate + Model Gate + Post-Process Gate + plan-edit child runs + before/after comparisons).

## Before vs After (Plan Edit) Outputs

When a plan edit creates a child run, the system also writes:
- `logs/runs/<run_id>/child_runs/<child_run_id>/reports/before_after_report.md`
- `logs/runs/<run_id>/child_runs/<child_run_id>/reports/before_after_overlay.png`
- `logs/runs/<run_id>/child_runs/<child_run_id>/reports/residual_diff_hist.png`
- `logs/runs/<run_id>/child_runs/<child_run_id>/reports/before_after_aligned.parquet`

# Phase 103: ReACT Reasoning Loop + Telemetry (What To Expect)

This phase adds **explicit reasoning telemetry** to the blackboard (`logs/message_pool.jsonl`) so the UI can render:
- Observation → Thought → Action cycles (ReACT)
- Hyperparameter importance after Optuna trials
- Token/cost telemetry for LLM usage
- Preparation steps for a Fig.4-like prep trace

## New Topics You Will See

### ModelManagerAgent (baseline / HPO)
- `observation/<run_id>`:
  - `what=data_ready` when preparation is ready
  - `what=trial_result` after an evaluation completes (RMSE/MAE)
- `thought/<run_id>`:
  - short natural-language summaries like “enqueue next trial”
- `action/<run_id>`:
  - `action=publish_train_command`
  - `action=enqueue_next_trial`
  - `action=stop_hpo` / `action=idle`
- `hyperparameter_importance/<run_id>`:
  - `importances={...}` from `optuna.importance.get_param_importances(study)` when enough trials exist

### PreparationAgent
- `prepare_step/<run_id>`:
  - `metadata_collected`
  - `validity_check_passed`
  - `columns_defined`
  - `cleaning_preferences_captured` (non-blocking; just logs any `system_directive.overrides` seen)

### InteractiveTaskManagerAgent (when `--use_llm` is enabled and LLM is actually called)
- `llm_usage/<run_id>`:
  - `usage={...}` best-effort token counts
  - `cost_usd` only if you set env rates
- `telemetry_cost/<run_id>`:
  - normalized cost event for dashboards

## How To View It

1. Run any baseline mode with HPO enabled (Optuna):
   - You should see `observation`, `thought`, `action`, and eventually `hyperparameter_importance`.
2. Run interactive research with `--use_llm` and give a free-form command that the rule parser can’t parse:
   - You should see `llm_usage` + `telemetry_cost`.
3. Open the React UI and go to:
   - **Agent Flow** tab: watch these topics appear as arrows between lifelines.

## Cost Configuration (Optional)

To get `cost_usd` in telemetry, set:
- `OPENAI_USD_PER_1K_INPUT_TOKENS`
- `OPENAI_USD_PER_1K_OUTPUT_TOKENS`

If not set, cost will be `null` but token usage will still be logged.


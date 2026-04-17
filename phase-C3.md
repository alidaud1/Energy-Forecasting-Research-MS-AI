# Phase C3 - Net-Load Validation, Thesis Visuals, and System Auditing

Phase C3 upgrades Phase C2 from "we can train transformers" to "research-grade reporting":
- Net-load validation (only when alignment exists) and net-load MAE/RMSE
- Thesis-ready plots (triple-axis week + baseline-vs-transformer error distributions)
- A lightweight audit appendix extracted from `logs/message_pool.jsonl`

## Key Changes

1. **Predictions artifacts are now standardized**

Both baselines and transformers write a long-format Parquet with these core columns:
- `run_id`, `track`, `model_family`, `model_name`
- `ZONEID`, `timestamp`, `horizon`
- `y_true`, `y_pred`

This enables correct alignment for net-load and distribution plots.

2. **Integration agent is research-grade**

`src/agents/forecast_integration_agent.py` now:
- collects multiple models per track
- selects best per track (prefers transformers) by RMSE/MAE
- computes net-load and its MAE/RMSE if overlap exists
- writes:
  - `logs/runs/<run_id>/reports/joint_decision_report.md`
  - `logs/runs/<run_id>/reports/final_research_summary.md`
  - `logs/runs/<run_id>/reports/system_performance_audit.md`
- writes plots under `logs/runs/<run_id>/plots/`

3. **Orchestrator can run baselines + transformers in one run**

In research mode, add `--with_baselines` to also run XGBoost baselines for load+solar (same `run_id`), enabling baseline-vs-transformer error plots.

## How To Run

Precheck Phase A2 artifacts:

```powershell
python scripts/phase_c_precheck.py --manifest data/processed/dataset_manifest.json
```

Transformers only (load + solar):

```powershell
python orchestrator.py --mode research --print_messages --timeout_seconds 7200 --epochs 5 --batch_size 128 --lr 0.001 --patience 2
```

Transformers + baselines (recommended for C3 visuals):

```powershell
python orchestrator.py --mode research --with_baselines --print_messages --timeout_seconds 7200
```

Export message trace appendix for a run:

```powershell
python scripts/export_message_pool.py --run_id run_YYYYMMDD_HHMMSS
```


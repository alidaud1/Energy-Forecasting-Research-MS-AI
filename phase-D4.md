# Phase D4 - Final Execution + Thesis-Style Audit (HITL)

This phase is about making the interactive system “thesis defensible”: the run should produce a clear record of (1) what the human said, (2) what structured directives were generated/applied, (3) the proposed vs human-adjusted variants, and (4) the metric impact when both variants exist.

## What You Run

1) Run an interactive research session (real dataset artifacts from `data/processed/`):

```powershell
python orchestrator.py --mode research --interactive --with_baselines --print_messages --timeout_seconds 7200 --epochs 5 --patience 2
```

2) After the run completes, build a standalone thesis-style audit markdown from the JSONL log:

```powershell
python scripts/interaction_audit.py --log logs/message_pool.jsonl --run_id <YOUR_RUN_ID> --out logs/runs/<YOUR_RUN_ID>/reports/interaction_audit_standalone.md
```

## Expected Outputs

- `logs/runs/<run_id>/reports/joint_decision_report.md`
- `logs/runs/<run_id>/reports/final_research_summary.md`
- `logs/runs/<run_id>/reports/interaction_audit.md` (written by the integration agent)
- `logs/runs/<run_id>/reports/interaction_audit_standalone.md` (written by `scripts/interaction_audit.py`)
- `logs/runs/<run_id>/reports/comparative_branch_report.md` (only when the human edits the plan; runs proposed + human_adjusted under one `comparison_group_id`)

## Notes

- If your task pair (load vs solar) has no overlapping timestamps/horizons, net-load cannot be computed; the reports will say so.
- Phase D3 dual-branch is the rule for making a claim like “human guidance improved metrics”: it evaluates both variants on the same splits/seed.


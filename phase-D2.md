# Phase D2 - Research-Grade Interactive Runtime + Audit

Phase D2 strengthens the HITL layer from Phase D1 so it behaves like a real interactive control plane:
- input does not need to block the whole system
- raw human language is translated in exactly one place (Task Manager)
- the run produces a thesis-ready `interaction_audit.md`
- interaction state is “memory-compressed” into compact summaries

## What’s Implemented

### 1) Non-blocking interactive input (default orchestrator)
`orchestrator.py --interactive` now uses a background stdin reader thread + queue, so training/status logs can continue while the user is typing.

Transformer runs can also be bounded on slower hardware via `--max_train_windows/--max_test_windows` (propagated into both baseline + transformer training payloads).

### 2) Task Manager is the translator boundary
Interactive mode publishes:
- raw: `user_feedback/<run_id>` (human_text)
- structured: `directive/<run_id>` (structured_command)

Other agents (notably the integration agent) consume **directives**, not raw English.

### 3) Memory compression
After each interaction, the task manager publishes:
- `state/<run_id>` event with a compact `summary` object (approved gates, active plan, last directive).

### 4) Interaction audit report
The integration agent writes:
- `logs/runs/<run_id>/reports/interaction_audit.md`

It includes gates, raw inputs, parsed directives, evaluation results, and links to plan/postprocess comparison artifacts.

## Phase D3 Add-On (Comparative Evidence)
If the human edits the model plan, the system will run **both**:
- `variant=proposed` (original plan snapshot)
- `variant=human_adjusted` (edited plan)

Both branches share the same `comparison_group_id` and are saved under `logs/runs/<run_id>/child_runs/`.
A comparative summary is written to `logs/runs/<run_id>/reports/comparative_branch_report.md`.

### 5) Optional fully-async interactive entrypoint
For a cleaner terminal experience, use:

```powershell
python orchestrator_async.py --with_baselines --print_messages
```

If `prompt_toolkit` is installed, it uses `prompt_async()` + `patch_stdout()`; otherwise it falls back to a safe thread-based stdin read.

## Notes

- This phase keeps your real pipeline unchanged (data preparation, training, integration).
- The interactive layer is a control plane on top of the message pool.

from __future__ import annotations

import json
from pathlib import Path

from scripts.interaction_audit import build_markdown


def test_interaction_audit_builds_markdown(tmp_path: Path) -> None:
    run_id = "run_audit_test"
    msgs = [
        {"run_id": run_id, "topic": f"system/{run_id}", "payload": {"event": "run_started", "plan": {"x": 1}}},
        {"run_id": run_id, "topic": f"user_feedback/{run_id}", "payload": {"stage": "prep", "human_text": "approve"}},
        {"run_id": run_id, "topic": f"system_directive/{run_id}", "seq": 1, "payload": {"stage": "model", "overrides": {"plan.epochs": 2}}},
        {"run_id": run_id, "topic": f"directive_ack/{run_id}", "payload": {"stage": "training", "applied_overrides": {"epochs": 2}}},
        {"run_id": run_id, "topic": f"directive_rejected/{run_id}", "payload": {"stage": "training", "rejected_overrides": {"plan.foo": 1}}},
        {
            "run_id": run_id,
            "topic": f"evaluate/{run_id}",
            "payload": {"event": "evaluation_complete", "track": "load", "variant": "proposed", "metrics": {"mae": 10.0, "rmse": 12.0}},
        },
        {
            "run_id": run_id,
            "topic": f"evaluate/{run_id}",
            "payload": {"event": "evaluation_complete", "track": "load", "variant": "human_adjusted", "metrics": {"mae": 9.0, "rmse": 11.0}},
        },
    ]
    md = build_markdown(msgs, run_id=run_id)
    assert "Interaction Audit" in md
    assert "Human interventions" in md
    assert "Metric comparison" in md
    assert "Impact of human guidance (load)" in md
    assert "Rejected overrides" in md

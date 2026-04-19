from __future__ import annotations

from src.api.state_projection import latest_active_gate


def test_latest_active_gate_picks_latest_unresolved():
    run_id = "run_x"
    msgs = [
        {"run_id": run_id, "topic": f"gate/{run_id}", "seq": 1, "correlation_id": "c1", "payload": {"event": "gate_requested", "gate_id": "g1", "stage": "prep", "prompt": "p1"}},
        {"run_id": run_id, "topic": f"gate/{run_id}", "seq": 2, "correlation_id": "c1", "payload": {"event": "gate_approved", "gate_id": "g1", "stage": "prep"}},
        {"run_id": run_id, "topic": f"gate/{run_id}", "seq": 3, "correlation_id": "c2", "payload": {"event": "gate_requested", "gate_id": "g2", "stage": "model", "prompt": "p2"}},
    ]
    g = latest_active_gate(msgs, run_id=run_id)
    assert g is not None
    assert g.gate_id == "g2"
    assert g.stage == "model"
    assert g.prompt == "p2"


from __future__ import annotations

from src.agents.base_agent import AgentConfig
from src.agents.interactive_task_manager_agent import InteractiveResearchConfig, InteractiveTaskManagerAgent
from src.agents.message_pool import MessagePool
from src.agents.messages import Message, TopicId


def _ready_payload(track: str) -> dict:
    return {
        "event": "ready",
        "track": track,
        "manifest_path": "data/processed/dataset_manifest.json",
        "paths": {
            "clean": "x",
            "train": "x",
            "val": "x",
            "test": "x",
            "train_scaled": "x",
            "val_scaled": "x",
            "test_scaled": "x",
            "scaler": "x",
        },
        "columns": {
            "feature_cols": ["f1"],
            "target_cols": ["y"],
            "hist_cols": ["y", "f1"],
            "future_cols": ["f1"],
        },
        "summary": {
            "train_scaled": {"n_rows": 10, "n_zones": 1, "ts_min": "2000-01-01", "ts_max": "2000-01-02", "na_target": 0},
            "val_scaled": {"n_rows": 10, "n_zones": 1, "ts_min": "2000-01-03", "ts_max": "2000-01-04", "na_target": 0},
            "test_scaled": {"n_rows": 10, "n_zones": 1, "ts_min": "2000-01-05", "ts_max": "2000-01-06", "na_target": 0},
        },
        "run_config": {"interactive": True},
    }


def test_phase_d3_dual_branch_publishes_proposed_and_adjusted(tmp_path):
    run_id = "run_test_d3"
    pool = MessagePool(log_path=str(tmp_path / "pool.jsonl"), load_existing=False)
    agent = InteractiveTaskManagerAgent(
        AgentConfig(agent_id="InteractiveTaskManagerAgent", role="task_manager", run_id=run_id),
        pool,
        InteractiveResearchConfig(tracks=["load", "solar"], with_baselines=True),
    )

    # Boot: publishes system + prepare commands.
    agent.tick()

    # Simulate preparation "ready" events for both tracks.
    pool.publish(
        Message.new(
            topic=TopicId("prepare", run_id),
            kind="event",
            sender="PreparationAgent",
            sender_role="preparation",
            payload=_ready_payload("load"),
            run_id=run_id,
        )
    )
    pool.publish(
        Message.new(
            topic=TopicId("prepare", run_id),
            kind="event",
            sender="PreparationAgent",
            sender_role="preparation",
            payload=_ready_payload("solar"),
            run_id=run_id,
        )
    )
    agent.tick()

    # Approve prep gate.
    pool.publish(
        Message.new(
            topic=TopicId("user_feedback", run_id),
            kind="event",
            sender="Human",
            sender_role="human",
            payload={"gate_id": agent.GATE_ID_PREP, "human_text": "approve"},
            run_id=run_id,
        )
    )
    agent.tick()

    # Edit plan at model gate, then approve.
    pool.publish(
        Message.new(
            topic=TopicId("user_feedback", run_id),
            kind="event",
            sender="Human",
            sender_role="human",
            payload={"gate_id": agent.GATE_ID_MODEL, "human_text": "set epochs=2"},
            run_id=run_id,
        )
    )
    agent.tick()

    pool.publish(
        Message.new(
            topic=TopicId("user_feedback", run_id),
            kind="event",
            sender="Human",
            sender_role="human",
            payload={"gate_id": agent.GATE_ID_MODEL, "human_text": "approve"},
            run_id=run_id,
        )
    )
    agent.tick()

    trains = [m for m in pool.find_by_topic("train", topic_source=run_id) if m.kind == "command"]
    variants = {str((m.payload or {}).get("variant") or "") for m in trains}
    assert "proposed" in variants
    assert "human_adjusted" in variants

    cmp_ids = {str((m.payload or {}).get("comparison_group_id") or "") for m in trains}
    cmp_ids.discard("")
    assert len(cmp_ids) == 1

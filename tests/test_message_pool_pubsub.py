import json
from pathlib import Path

from src.agents.message_pool import MessagePool
from src.agents.messages import Message, TopicId


def test_pubsub_topic_and_source_scoping(tmp_path: Path):
    log_path = tmp_path / "message_pool.jsonl"
    pool = MessagePool(str(log_path), load_existing=False)

    run_id = "run_2026_04_16_001"
    other_run = "run_2026_04_16_002"

    pool.subscribe("PrepAgent", "preparation", topics=["prepare"], topic_source=run_id, start_from="earliest")

    pool.publish(
        Message.new(
            topic=TopicId("prepare", run_id),
            kind="command",
            sender="TaskManagerAgent",
            sender_role="task_manager",
            payload={"x": 1},
        )
    )
    pool.publish(
        Message.new(
            topic=TopicId("prepare", other_run),
            kind="command",
            sender="TaskManagerAgent",
            sender_role="task_manager",
            payload={"x": 2},
        )
    )

    msgs = pool.poll("PrepAgent")
    assert len(msgs) == 1
    assert msgs[0].payload["x"] == 1


def test_directed_delivery_by_role_and_agent(tmp_path: Path):
    log_path = tmp_path / "message_pool.jsonl"
    pool = MessagePool(str(log_path), load_existing=False)

    pool.subscribe("A1", "role_a", topics=["train"], start_from="earliest")
    pool.subscribe("B1", "role_b", topics=["train"], start_from="earliest")

    pool.publish(
        Message.new(
            topic=TopicId("train", "r1"),
            kind="event",
            sender="TaskManagerAgent",
            sender_role="task_manager",
            to_roles=["role_a"],
            payload={"m": "to_role_a"},
        )
    )
    pool.publish(
        Message.new(
            topic=TopicId("train", "r1"),
            kind="event",
            sender="TaskManagerAgent",
            sender_role="task_manager",
            to=["B1"],
            payload={"m": "to_B1"},
        )
    )

    a_msgs = pool.poll("A1")
    b_msgs = pool.poll("B1")
    assert [m.payload["m"] for m in a_msgs] == ["to_role_a"]
    assert [m.payload["m"] for m in b_msgs] == ["to_B1"]


def test_cursor_advances_and_no_self_echo(tmp_path: Path):
    log_path = tmp_path / "message_pool.jsonl"
    pool = MessagePool(str(log_path), load_existing=False)

    pool.subscribe("AgentX", "role_x", topics=["system"], start_from="earliest")

    # self-echo should not be delivered
    pool.publish(
        Message.new(
            topic=TopicId("system", "r1"),
            kind="event",
            sender="AgentX",
            sender_role="role_x",
            payload={"n": 1},
        )
    )
    assert pool.poll("AgentX") == []

    pool.publish(
        Message.new(
            topic=TopicId("system", "r1"),
            kind="event",
            sender="Other",
            sender_role="other",
            payload={"n": 2},
        )
    )
    msgs1 = pool.poll("AgentX")
    assert len(msgs1) == 1
    assert msgs1[0].payload["n"] == 2

    # cursor advanced: no new messages now
    assert pool.poll("AgentX") == []


def test_jsonl_persistence_and_reload(tmp_path: Path):
    log_path = tmp_path / "message_pool.jsonl"
    pool = MessagePool(str(log_path), load_existing=False)
    pool.subscribe("R", "r", topics=["*"], start_from="earliest")

    m = pool.publish(
        Message.new(
            topic=TopicId("prepare", "run1"),
            kind="command",
            sender="TaskManagerAgent",
            sender_role="task_manager",
            correlation_id="corr1",
            payload={"p": 1},
        )
    )
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["msg_id"] == m.msg_id

    pool2 = MessagePool(str(log_path), load_existing=True)
    assert pool2.size() == 1
    found = pool2.find_by_correlation_id("corr1")
    assert len(found) == 1
    assert found[0].payload["p"] == 1


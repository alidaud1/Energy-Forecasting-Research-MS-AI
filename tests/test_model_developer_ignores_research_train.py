from pathlib import Path

from src.agents.base_agent import AgentConfig
from src.agents.message_pool import MessagePool
from src.agents.messages import Message, TopicId
from src.agents.model_developer_agent import ModelDeveloperAgent


def test_model_developer_ignores_train_without_model_type(tmp_path: Path, monkeypatch):
    # Avoid writing into the real repo logs/ during test runs.
    monkeypatch.chdir(tmp_path)

    run_id = "run_test_research_train_ignore"
    pool = MessagePool(str(tmp_path / "message_pool.jsonl"), load_existing=False)

    dev = ModelDeveloperAgent(AgentConfig(agent_id="ModelDeveloperAgent", role="model_developer", run_id=run_id), pool)

    # Research-mode "train" plan messages do not include model_type and must be ignored here
    # (they are converted to train_execute by forecasting agents).
    pool.publish(
        Message.new(
            topic=TopicId("train", run_id),
            kind="command",
            sender="Orchestrator",
            sender_role="system",
            payload={"command": "train", "track": "load"},
        )
    )

    dev.tick()

    eval_msgs = pool.find_by_topic("evaluate", topic_source=run_id)
    assert eval_msgs == []


from __future__ import annotations

import uuid
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentConfig


@dataclass
class RunConfig:
    manifest_path: str
    track: str  # "load" | "solar"
    model_type: str  # "linear" | "xgboost"
    window: dict  # {"W": 168, "H": 24, "stride": 24}
    training: dict  # {"max_train_windows": 20000, "max_test_windows": 5000, "seed": 42}
    hpo: dict  # {"enabled": False, "n_trials": 1, ...}


class TaskManagerAgent(BaseAgent):
    """
    Starts a run by publishing:
    - a system event with run metadata
    - a prepare command with manifest + track + run_config
    """

    def __init__(self, cfg: AgentConfig, pool, run_cfg: RunConfig):
        super().__init__(cfg, pool)
        self.run_cfg = run_cfg
        self._started = False
        self.subscribe(topics=["system"], start_from="latest")

    def tick(self) -> None:
        if self._started:
            return
        self._started = True

        corr_id = str(uuid.uuid4())

        run_config_payload = {
            "track": self.run_cfg.track,
            "model_type": self.run_cfg.model_type,
            "window": self.run_cfg.window,
            "training": self.run_cfg.training,
            "hpo": self.run_cfg.hpo,
        }

        self.publish(
            topic_type="system",
            kind="event",
            payload={"event": "run_started", **run_config_payload},
            correlation_id=corr_id,
        )

        self.publish(
            topic_type="prepare",
            kind="command",
            payload={
                "command": "prepare",
                "manifest_path": self.run_cfg.manifest_path,
                "track": self.run_cfg.track,
                "run_config": run_config_payload,
            },
            correlation_id=corr_id,
        )


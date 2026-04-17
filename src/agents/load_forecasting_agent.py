from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base_agent import BaseAgent


@dataclass
class LoadTransformerDefaults:
    model_name: str = "patchtst"  # "patchtst" | "itransformer"
    strategy: str = "A"  # load often OK with history-only
    add_future_indicator: bool = True
    use_hour_embedding: bool = False


class LoadForecastingAgent(BaseAgent):
    """
    Specializes transformer execution config for load.
    Converts a `train` command into a directed `train_execute` command for ModelDeveloper.
    """

    def __init__(self, cfg, pool, defaults: Optional[LoadTransformerDefaults] = None):
        super().__init__(cfg, pool)
        self.defaults = defaults or LoadTransformerDefaults()
        self.subscribe(topics=["train"], start_from="latest")

    def tick(self) -> None:
        for msg in self.poll(max_messages=50):
            if msg.kind != "command":
                continue
            if msg.payload.get("command") != "train":
                continue
            if msg.payload.get("track") != "load":
                continue

            corr_id = msg.correlation_id
            plan = msg.payload

            exec_cfg = {
                "model_name": plan.get("model_name", self.defaults.model_name),
                "strategy": plan.get("strategy", self.defaults.strategy),
                "add_future_indicator": bool(plan.get("add_future_indicator", self.defaults.add_future_indicator)),
                "use_hour_embedding": bool(plan.get("use_hour_embedding", self.defaults.use_hour_embedding)),
            }

            self.publish(
                topic_type="train_execute",
                kind="command",
                payload={
                    "command": "train_execute",
                    "track": "load",
                    "paths": plan["paths"],
                    "columns": plan["columns"],
                    "window": plan["window"],
                    "training": plan.get("training", {}),
                    "exec_cfg": exec_cfg,
                },
                correlation_id=corr_id,
                to_roles=["model_developer"],
            )


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .base_agent import BaseAgent


SOLAR_ACCUM_COLS = ["VAR169", "VAR175", "VAR178", "VAR228"]


@dataclass
class SolarTransformerDefaults:
    model_name: str = "patchtst"
    strategy: str = "B"  # solar benefits from known-future covariates
    add_future_indicator: bool = True
    use_hour_embedding: bool = True
    selected_predictors: Optional[List[str]] = None


class SolarForecastingAgent(BaseAgent):
    """
    Solar specialization:
    - optionally reduces predictors used (selected_predictors)
    - asserts accumulated vars exist (sanity check)
    - converts `train` command into directed `train_execute` for ModelDeveloper
    """

    def __init__(self, cfg, pool, defaults: Optional[SolarTransformerDefaults] = None):
        super().__init__(cfg, pool)
        self.defaults = defaults or SolarTransformerDefaults()
        self.subscribe(topics=["train"], start_from="latest")

    def _select_predictors(self, columns: dict, selected_predictors: Optional[List[str]]) -> dict:
        if not selected_predictors:
            return columns

        keep = set(selected_predictors)
        hist_cols = columns["hist_cols"]
        fut_cols = columns["future_cols"]

        def ok(c: str) -> bool:
            return c == "POWER" or c.startswith("hour_") or c.startswith("dow_") or c.startswith("month_") or (c in keep)

        new_hist = [c for c in hist_cols if ok(c)]
        new_fut = [c for c in fut_cols if (c.startswith("hour_") or c.startswith("dow_") or c.startswith("month_")) or (c in keep)]

        out = dict(columns)
        out["hist_cols"] = new_hist
        out["future_cols"] = new_fut
        return out

    def _assert_vars_present(self, df_train_scaled_path: str, required: List[str]) -> None:
        df = pd.read_parquet(df_train_scaled_path, columns=[c for c in required if c])
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Solar required predictor '{c}' not present in processed data.")

    def tick(self) -> None:
        for msg in self.poll(max_messages=50):
            if msg.kind != "command":
                continue
            if msg.payload.get("command") != "train":
                continue
            if msg.payload.get("track") != "solar":
                continue

            corr_id = msg.correlation_id
            plan = msg.payload

            self._assert_vars_present(plan["paths"]["train_scaled"], SOLAR_ACCUM_COLS)

            cols = self._select_predictors(
                plan["columns"],
                plan.get("selected_predictors", self.defaults.selected_predictors),
            )

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
                    "track": "solar",
                    "paths": plan["paths"],
                    "columns": cols,
                    "window": plan["window"],
                    "training": plan.get("training", {}),
                    "exec_cfg": exec_cfg,
                },
                correlation_id=corr_id,
                to_roles=["model_developer"],
            )


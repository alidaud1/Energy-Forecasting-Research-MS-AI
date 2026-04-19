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
        # Phase D3: also listen for structured directives (control-plane messages).
        self.subscribe(topics=["train", "system_directive"], start_from="latest")
        self._applied_directives: set[str] = set()
        self._training_overrides: dict = {}

    def _apply_overrides(self, overrides: dict) -> tuple[dict, dict]:
        # Keep it intentionally small/safe: only allow overrides that map to existing training/window knobs.
        applied: dict = {}
        rejected: dict = {}
        for k, v in (overrides or {}).items():
            ks = str(k)
            if ks in ("optimizer.lr", "plan.lr"):
                try:
                    val = float(v)
                    self._training_overrides["lr"] = val
                    applied[ks] = val
                except Exception:
                    rejected[ks] = v
            elif ks in ("trainer.max_epochs", "plan.epochs"):
                try:
                    val = int(v)
                    self._training_overrides["epochs"] = val
                    applied[ks] = val
                except Exception:
                    rejected[ks] = v
            elif ks == "plan.batch_size":
                try:
                    val = int(v)
                    self._training_overrides["batch_size"] = val
                    applied[ks] = val
                except Exception:
                    rejected[ks] = v
            elif ks in ("window.W", "plan.W"):
                try:
                    val = int(v)
                    self._training_overrides["W"] = val
                    applied[ks] = val
                except Exception:
                    rejected[ks] = v
            elif ks in ("window.H", "plan.H"):
                try:
                    val = int(v)
                    self._training_overrides["H"] = val
                    applied[ks] = val
                except Exception:
                    rejected[ks] = v
            elif ks in ("window.stride", "plan.stride"):
                try:
                    val = int(v)
                    self._training_overrides["stride"] = val
                    applied[ks] = val
                except Exception:
                    rejected[ks] = v
            else:
                rejected[ks] = v
        return applied, rejected

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
            if msg.kind == "event" and msg.topic.startswith(f"system_directive/{self.run_id}"):
                if msg.msg_id in self._applied_directives:
                    continue
                payload = msg.payload or {}
                stage = payload.get("stage")
                if stage not in ("training", "model", "prep", "global", None, ""):
                    continue
                target_roles = set(payload.get("target_roles") or ["*"])
                if "*" not in target_roles and self.role not in target_roles:
                    continue
                overrides = payload.get("overrides") or {}
                before = dict(self._training_overrides)
                applied, rejected = self._apply_overrides(overrides if isinstance(overrides, dict) else {})
                after = dict(self._training_overrides)
                self._applied_directives.add(msg.msg_id)
                if before != after:
                    self.publish(
                        topic_type="directive_ack",
                        kind="event",
                        payload={
                            "event": "directive_ack",
                            "stage": "training",
                            "directive_id": msg.msg_id,
                            "applied_overrides": after,
                            "applied_keys": sorted(applied.keys()),
                        },
                        correlation_id=msg.correlation_id,
                    )
                elif rejected:
                    self.publish(
                        topic_type="directive_rejected",
                        kind="event",
                        payload={
                            "event": "directive_rejected",
                            "stage": "training",
                            "directive_id": msg.msg_id,
                            "rejected_overrides": rejected,
                        },
                        correlation_id=msg.correlation_id,
                        level="warning",
                    )
                continue

            if msg.kind != "command":
                continue
            if msg.payload.get("command") != "train":
                continue
            if msg.payload.get("track") != "solar":
                continue

            corr_id = msg.correlation_id
            plan = msg.payload

            self._assert_vars_present(plan["paths"]["train_scaled"], SOLAR_ACCUM_COLS)

            # Apply any accumulated overrides from system_directive.
            training = dict(plan.get("training", {}) or {})
            for k in ("lr", "epochs", "batch_size", "patience", "max_train_windows", "max_val_windows", "max_test_windows"):
                if k in self._training_overrides:
                    training[k] = self._training_overrides[k]
            window = dict(plan.get("window", {}) or {})
            for k in ("W", "H", "stride"):
                if k in self._training_overrides:
                    window[k] = self._training_overrides[k]

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
                    "window": window,
                    "training": training,
                    "exec_cfg": exec_cfg,
                    "child_run_id": plan.get("child_run_id", ""),
                    "parent_run_id": plan.get("parent_run_id", ""),
                    "comparison_group_id": plan.get("comparison_group_id", ""),
                    "variant": plan.get("variant", ""),
                },
                correlation_id=corr_id,
                to_roles=["model_developer"],
            )

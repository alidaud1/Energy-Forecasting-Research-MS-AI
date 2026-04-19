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
            if msg.payload.get("track") != "load":
                continue

            corr_id = msg.correlation_id
            plan = msg.payload

            # Apply any accumulated overrides from system_directive.
            training = dict(plan.get("training", {}) or {})
            for k in ("lr", "epochs", "batch_size", "patience", "max_train_windows", "max_val_windows", "max_test_windows"):
                if k in self._training_overrides:
                    training[k] = self._training_overrides[k]
            window = dict(plan.get("window", {}) or {})
            for k in ("W", "H", "stride"):
                if k in self._training_overrides:
                    window[k] = self._training_overrides[k]

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

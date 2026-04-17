from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import optuna

from .base_agent import BaseAgent


@dataclass
class HPOState:
    enabled: bool
    n_trials: int
    completed: int = 0
    study_name: str = ""
    storage: Optional[str] = None
    current_trial_number: Optional[int] = None


class ModelManagerAgent(BaseAgent):
    """
    Receives 'ready' event and emits a 'train' command (with optional Optuna ask/tell skeleton).
    """

    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        self.subscribe(topics=["prepare", "evaluate"], start_from="latest")
        self.hpo_state: Optional[HPOState] = None
        self._study: Optional[optuna.Study] = None
        self._trial = None

    def _make_default_params(self, model_type: str) -> Dict[str, Any]:
        if model_type == "linear":
            return {"fit_intercept": True}
        if model_type == "xgboost":
            return {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "random_state": 42,
                "tree_method": "hist",
            }
        raise ValueError(f"Unsupported model_type: {model_type}")

    def _init_optuna(self, base_config: dict) -> None:
        hpo = base_config.get("hpo", {}) or {}
        enabled = bool(hpo.get("enabled", False))
        n_trials = int(hpo.get("n_trials", 1))
        self.hpo_state = HPOState(enabled=enabled, n_trials=n_trials)

        if not enabled:
            return

        storage = hpo.get("storage", None)
        study_name = hpo.get("study_name", f"study_{self.run_id}")

        self._study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        self.hpo_state.storage = storage
        self.hpo_state.study_name = study_name

    def _sample_params_optuna(self, model_type: str, default_params: dict) -> dict:
        if not self._study:
            return default_params

        self._trial = self._study.ask()
        if self.hpo_state:
            self.hpo_state.current_trial_number = self._trial.number

        if model_type == "xgboost":
            p = dict(default_params)
            p["max_depth"] = self._trial.suggest_int("max_depth", 3, 10)
            p["learning_rate"] = self._trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            p["subsample"] = self._trial.suggest_float("subsample", 0.5, 1.0)
            p["colsample_bytree"] = self._trial.suggest_float("colsample_bytree", 0.5, 1.0)
            p["reg_lambda"] = self._trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            p["n_estimators"] = self._trial.suggest_int("n_estimators", 100, 800)
            return p

        return default_params

    def tick(self) -> None:
        for msg in self.poll(max_messages=50):
            if msg.topic.startswith(f"prepare/{self.run_id}") and msg.kind == "event" and msg.payload.get("event") == "ready":
                corr_id = msg.correlation_id
                base_cfg = msg.payload
                run_cfg = base_cfg.get("run_config", {}) or {}

                track = base_cfg["track"]
                model_type = run_cfg.get("model_type", "xgboost")
                window = run_cfg.get("window", {"W": 168, "H": 24, "stride": 24})
                training = run_cfg.get("training", {"max_train_windows": 20000, "max_test_windows": 5000, "seed": 42})
                hpo = run_cfg.get("hpo", {"enabled": False, "n_trials": 1})

                plan = {
                    "command": "train",
                    "track": track,
                    "paths": base_cfg["paths"],
                    "columns": base_cfg["columns"],
                    "model_type": model_type,
                    "hyperparams": self._make_default_params(model_type),
                    "window": window,
                    "training": training,
                    "hpo": hpo,
                }

                self._init_optuna(plan)
                if self.hpo_state and self.hpo_state.enabled:
                    plan["hyperparams"] = self._sample_params_optuna(model_type, plan["hyperparams"])
                    plan["optuna"] = {
                        "enabled": True,
                        "study_name": self.hpo_state.study_name,
                        "storage": self.hpo_state.storage,
                        "trial_number": self.hpo_state.current_trial_number,
                    }

                self.publish(topic_type="train", kind="command", payload=plan, correlation_id=corr_id)

            if msg.topic.startswith(f"evaluate/{self.run_id}") and msg.kind == "result":
                if self.hpo_state and self.hpo_state.enabled and self._study and self._trial is not None:
                    rmse = float(msg.payload.get("metrics", {}).get("rmse", float("nan")))
                    if rmse == rmse:
                        self._study.tell(self._trial, rmse)
                        self.hpo_state.completed += 1


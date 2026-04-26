from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Dict, Optional, Tuple

import optuna
from optuna.importance import get_param_importances  # type: ignore
from optuna.samplers import TPESampler  # type: ignore

from .base_agent import BaseAgent


@dataclass
class HPOState:
    enabled: bool
    n_trials: int
    completed: int = 0
    study_name: str = ""
    storage: Optional[str] = None
    current_trial_number: Optional[int] = None
    init_batch_K: int = 1
    batch_B: int = 1
    epsilon_rmse: Optional[float] = None
    constant_liar: bool = False


class ModelManagerAgent(BaseAgent):
    """
    Phase 103: ReACT-style manager.

    Receives 'ready' events and manages baseline training/HPO via explicit:
    - observation (what we saw)
    - thought (why we think it matters)
    - action (what we will do next)

    It still publishes the concrete `train:command` messages used by ModelDeveloperAgent.
    """

    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        # Phase 104: listen to system_directive to apply interactive guidance to the search space.
        self.subscribe(topics=["prepare", "evaluate", "system_directive"], start_from="latest")
        self.hpo_state: Optional[HPOState] = None
        self._study: Optional[optuna.Study] = None
        self._trial = None
        self._base_cfg: Optional[dict] = None
        self._last_plan: Optional[dict] = None
        # Phase 104: batched ask/tell bookkeeping.
        self._batch_id: str = ""
        self._batch_trials: list[int] = []
        self._batch_results: Dict[int, float] = {}
        self._trial_by_number: Dict[int, optuna.trial.FrozenTrial] = {}  # best-effort (for audit only)
        self._asked_trials: Dict[int, optuna.trial.Trial] = {}
        # Phase 104: interactive guidance ranges for dynamic search space filtering.
        # param -> (low, high, is_int, log)
        self._guided_ranges: Dict[str, Tuple[float, float, bool, bool]] = {}

    def _make_default_params(self, model_type: str) -> Dict[str, Any]:
        mt = str(model_type or "").lower()
        if mt == "linear":
            # Table II: regularization + alpha (alpha only used for ridge).
            return {"fit_intercept": True, "regularization": "none", "alpha": 0.1}
        if mt == "svr":
            return {"C": 1.0, "gamma": 0.1}
        if mt == "mlp":
            # Table II: hidden_layers, hidden_size, learning_rate, dropout
            return {"hidden_layers": 2, "hidden_size": 64, "learning_rate": 1e-3, "dropout": 0.1}
        if mt in ("lstm", "gru"):
            # Table II: layers, hidden_size, fc_size, learning_rate
            return {"layers": 1, "hidden_size": 64, "fc_size": 64, "learning_rate": 1e-3}
        if mt == "cnn":
            # Table II: layers, kernel_size, filters, fc_size, learning_rate
            return {"layers": 1, "kernel_size": 3, "filters": 32, "fc_size": 64, "learning_rate": 1e-3}
        if mt == "xgboost":
            # Keep non-Table-II params fixed; only sample the Table II bounds when HPO is enabled.
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
        initK = int(hpo.get("init_batch_K", 1) or 1)
        batchB = int(hpo.get("batch_B", 1) or 1)
        const_liar = bool(hpo.get("constant_liar", False))
        eps = hpo.get("epsilon_rmse", None)
        try:
            eps_f = float(eps) if eps is not None else None
        except Exception:
            eps_f = None
        self.hpo_state = HPOState(enabled=enabled, n_trials=n_trials, init_batch_K=max(1, initK), batch_B=max(1, batchB), epsilon_rmse=eps_f, constant_liar=const_liar)

        if not enabled:
            return

        storage = hpo.get("storage", None)
        study_name = hpo.get("study_name", f"study_{self.run_id}")
        # Use a seeded TPE sampler so batched ask() can apply constant-liar diversification when enabled.
        # Seed comes from training config when available, else 42.
        seed = 42
        try:
            seed = int((base_config.get("training", {}) or {}).get("seed", 42))
        except Exception:
            seed = 42
        sampler = TPESampler(seed=seed, multivariate=True, constant_liar=bool(const_liar))

        self._study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
        )
        self.hpo_state.storage = storage
        self.hpo_state.study_name = study_name

    def _range(self, name: str, *, default: Tuple[float, float, bool, bool]) -> Tuple[float, float, bool, bool]:
        """
        Resolve guided range for a parameter if present; else default.
        Tuple: (low, high, is_int, log)
        """
        cur = self._guided_ranges.get(name)
        if not cur:
            return default
        low, high, is_int, log = cur
        dlow, dhigh, dis_int, dlog = default
        # If guidance only provided one bound, fall back to default for the missing side.
        if not (low == low):  # NaN
            low = dlow
        if not (high == high):  # NaN
            high = dhigh
        # Preserve typing/log defaults (guidance only controls bounds).
        is_int = dis_int
        log = dlog
        return (float(low), float(high), bool(is_int), bool(log))

    def _apply_guidance_override(self, overrides: Dict[str, Any]) -> None:
        """
        Phase 104: interpret system_directive overrides as dynamic search-space filters.
        Supported keys:
          hpo.<param>.min / hpo.<param>.max
        Example:
          hpo.learning_rate.min=0.01, hpo.learning_rate.max=0.1
        """
        for k, v in (overrides or {}).items():
            ks = str(k)
            if not ks.startswith("hpo.") or (".min" not in ks and ".max" not in ks):
                continue
            try:
                parts = ks.split(".")
                # ["hpo", "<param>", "min|max"]
                if len(parts) < 3:
                    continue
                param = parts[1]
                which = parts[2]
                cur = self._guided_ranges.get(param, (float("nan"), float("nan"), False, False))
                low, high, is_int, log = cur
                val = float(v)
                if which == "min":
                    low = val
                elif which == "max":
                    high = val
                # Preserve known typing/log defaults when we have them later; store raw now.
                self._guided_ranges[param] = (low, high, is_int, log)
            except Exception:
                continue

    def _sample_params_optuna(self, model_type: str, default_params: dict) -> dict:
        if not self._study:
            return default_params

        self._trial = self._study.ask()
        if self.hpo_state:
            self.hpo_state.current_trial_number = self._trial.number

        mt = str(model_type or "").lower()

        # Table II search spaces (strict bounds/steps).
        if mt == "linear":
            p = dict(default_params)
            p["regularization"] = self._trial.suggest_categorical("regularization", ["none", "ridge"])
            p["alpha"] = self._trial.suggest_float("alpha", 1e-4, 1.0, log=True)
            return p

        if mt == "svr":
            p = dict(default_params)
            p["C"] = self._trial.suggest_float("C", 1e-3, 1e3, log=True)
            p["gamma"] = self._trial.suggest_float("gamma", 1e-3, 1e3, log=True)
            return p

        if mt == "mlp":
            p = dict(default_params)
            p["hidden_layers"] = self._trial.suggest_int("hidden_layers", 2, 5, step=1)
            p["hidden_size"] = self._trial.suggest_int("hidden_size", 16, 512, log=True)
            p["learning_rate"] = self._trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            p["dropout"] = self._trial.suggest_float("dropout", 0.0, 0.5)
            return p

        if mt in ("lstm", "gru"):
            p = dict(default_params)
            p["layers"] = self._trial.suggest_int("layers", 1, 3, step=1)
            p["hidden_size"] = self._trial.suggest_int("hidden_size", 16, 512, log=True)
            p["fc_size"] = self._trial.suggest_int("fc_size", 16, 512, log=True)
            p["learning_rate"] = self._trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            return p

        if mt == "cnn":
            p = dict(default_params)
            p["layers"] = self._trial.suggest_int("layers", 1, 3, step=1)
            p["kernel_size"] = self._trial.suggest_int("kernel_size", 1, 5, step=1)
            p["filters"] = self._trial.suggest_int("filters", 16, 128, log=True)
            p["fc_size"] = self._trial.suggest_int("fc_size", 16, 512, log=True)
            p["learning_rate"] = self._trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            return p

        if mt == "xgboost":
            # Strict Table II bounds + step sizes.
            p = dict(default_params)
            p["n_estimators"] = self._trial.suggest_int("n_estimators", 10, 300, step=10)
            p["max_depth"] = self._trial.suggest_int("max_depth", 4, 16, step=2)
            p["learning_rate"] = self._trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            return p

        return default_params

    def _start_batch(self, *, plan_template: dict, batch_size: int, track: str) -> None:
        """
        Phase 104: issue a batch of ask() trials and publish train commands.
        We do not call tell() until all results for the batch are received.
        """
        if not self.hpo_state or not self.hpo_state.enabled or not self._study:
            return
        self._batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        self._batch_trials = []
        self._batch_results = {}

        self.publish(
            topic_type="observation",
            kind="event",
            payload={"event": "observation", "what": "batch_asked", "track": track, "batch_id": self._batch_id, "batch_size": int(batch_size)},
        )
        self.publish(
            topic_type="thought",
            kind="event",
            payload={"event": "thought", "track": track, "summary": f"Asking {batch_size} trials via Optuna ask-and-tell (constant_liar={self.hpo_state.constant_liar})."},
        )

        # Ask trials (batched).
        for _ in range(int(batch_size)):
            trial_params = self._sample_params_optuna(str(plan_template.get("model_type") or "xgboost"), dict(plan_template.get("hyperparams", {}) or {}))
            trial_number = int(self.hpo_state.current_trial_number or -1)
            if trial_number < 0:
                continue
            self._asked_trials[trial_number] = self._trial  # type: ignore[assignment]
            self._batch_trials.append(trial_number)
            plan = dict(plan_template)
            plan["hyperparams"] = dict(trial_params)
            plan["optuna"] = {
                "enabled": True,
                "study_name": self.hpo_state.study_name,
                "storage": self.hpo_state.storage,
                "trial_number": trial_number,
                "batch_id": self._batch_id,
            }
            corr_id = str(uuid.uuid4())
            self.publish(
                topic_type="action",
                kind="event",
                payload={"event": "action", "track": track, "action": "publish_train_command", "trial_number": trial_number, "batch_id": self._batch_id},
                correlation_id=corr_id,
            )
            self.publish(topic_type="train", kind="command", payload=plan, correlation_id=corr_id)

    def tick(self) -> None:
        for msg in self.poll(max_messages=50):
            # Phase 104: accept dynamic guidance to filter the search space for subsequent trials.
            if msg.kind == "event" and msg.topic.startswith(f"system_directive/{self.run_id}"):
                p = msg.payload or {}
                overrides = p.get("overrides") or {}
                if isinstance(overrides, dict):
                    self._apply_guidance_override(overrides)
                continue

            if msg.topic.startswith(f"prepare/{self.run_id}") and msg.kind == "event" and msg.payload.get("event") == "ready":
                base_cfg = msg.payload
                run_cfg = base_cfg.get("run_config", {}) or {}

                track = base_cfg["track"]
                model_type = run_cfg.get("model_type", "xgboost")
                window = run_cfg.get("window", {"W": 168, "H": 24, "stride": 24})
                training = run_cfg.get("training", {"max_train_windows": 20000, "max_test_windows": 5000, "seed": 42})
                hpo = run_cfg.get("hpo", {"enabled": False, "n_trials": 1})

                # Cache so we can schedule additional trials after evaluation results arrive.
                self._base_cfg = base_cfg

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
                # ReACT telemetry (Observation -> Thought -> Action)
                obs = {
                    "event": "observation",
                    "what": "data_ready",
                    "track": track,
                    "model_type": model_type,
                    "hpo_enabled": bool(self.hpo_state.enabled) if self.hpo_state else False,
                    "completed_trials": int(self.hpo_state.completed) if self.hpo_state else 0,
                }
                if self._study is not None and self._study.best_trial is not None:
                    try:
                        obs["best_value"] = float(self._study.best_value)
                        obs["best_trial"] = int(self._study.best_trial.number)
                    except Exception:
                        pass
                self.publish(topic_type="observation", kind="event", payload=obs)

                self._last_plan = dict(plan)
                if self.hpo_state and self.hpo_state.enabled:
                    # Initial batch (K)
                    self._start_batch(plan_template=plan, batch_size=int(self.hpo_state.init_batch_K), track=track)
                else:
                    # Non-HPO one-shot train
                    self.publish(
                        topic_type="thought",
                        kind="event",
                        payload={"event": "thought", "track": track, "summary": "Non-HPO: publish single train command."},
                    )
                    corr_id = str(uuid.uuid4())
                    self.publish(topic_type="action", kind="event", payload={"event": "action", "track": track, "action": "publish_train_command"}, correlation_id=corr_id)
                    self.publish(topic_type="train", kind="command", payload=plan, correlation_id=corr_id)

            if msg.topic.startswith(f"evaluate/{self.run_id}") and msg.kind == "result":
                metrics = msg.payload.get("metrics", {}) or {}
                rmse = float(metrics.get("rmse", float("nan")))
                mae = float(metrics.get("mae", float("nan")))
                track = str(msg.payload.get("track") or "")
                opt = msg.payload.get("optuna", {}) or {}
                try:
                    trial_number = int(opt.get("trial_number")) if opt.get("trial_number") is not None else None
                except Exception:
                    trial_number = None

                # Observation telemetry for the completed evaluation.
                self.publish(
                    topic_type="observation",
                    kind="event",
                    payload={"event": "observation", "what": "trial_result", "track": track, "rmse": rmse, "mae": mae, "trial_number": trial_number, "optuna": opt},
                    correlation_id=msg.correlation_id,
                )

                if self.hpo_state and self.hpo_state.enabled and self._study and self._trial is not None:
                    if rmse == rmse and trial_number is not None and trial_number in self._batch_trials:
                        # Batch-mode: store result, and only tell() once the full batch is complete.
                        self._batch_results[int(trial_number)] = float(rmse)
                        if len(self._batch_results) < len(self._batch_trials):
                            continue

                        # Tell all batch results.
                        for tn in list(self._batch_trials):
                            val = float(self._batch_results.get(tn, float("nan")))
                            if val == val:
                                try:
                                    self._study.tell(tn, val)
                                    self.hpo_state.completed += 1
                                except Exception:
                                    pass

                        # Surrogate update logging (post batch tell)
                        self.publish(
                            topic_type="thought",
                            kind="event",
                            payload={"event": "thought", "track": track, "summary": f"Batch complete (batch_id={opt.get('batch_id','')}). Surrogate updated with {len(self._batch_trials)} results."},
                        )
                        self.publish(
                            topic_type="action",
                            kind="event",
                            payload={"event": "action", "track": track, "action": "surrogate_updated", "batch_size": int(len(self._batch_trials))},
                            correlation_id=msg.correlation_id,
                        )

                        # Termination check: Tr or epsilon
                        eps = self.hpo_state.epsilon_rmse
                        try:
                            best = float(self._study.best_value) if self._study.best_trial is not None else float("inf")
                        except Exception:
                            best = float("inf")
                        if self.hpo_state.completed >= self.hpo_state.n_trials or (eps is not None and best <= float(eps)):
                            reason = "max_trials" if self.hpo_state.completed >= self.hpo_state.n_trials else "epsilon_reached"
                            self.publish(
                                topic_type="thought",
                                kind="event",
                                payload={"event": "thought", "track": track, "summary": f"Stopping HPO ({reason}). completed={self.hpo_state.completed} best_rmse={best} epsilon={eps}"},
                            )
                            self.publish(topic_type="action", kind="event", payload={"event": "action", "track": track, "action": "stop_hpo", "reason": reason})
                            # Clear batch state.
                            self._batch_trials = []
                            self._batch_results = {}
                            continue

                        # Publish hyperparameter importance when meaningful.
                        try:
                            if len([t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 2:
                                imp = get_param_importances(self._study)
                                self.publish(
                                    topic_type="hyperparameter_importance",
                                    kind="event",
                                    payload={
                                        "event": "hyperparameter_importance",
                                        "track": track,
                                        "study_name": self.hpo_state.study_name,
                                        "completed_trials": int(self.hpo_state.completed),
                                        "importances": {str(k): float(v) for k, v in imp.items()},
                                    },
                                )
                        except Exception:
                            pass

                        remaining = int(self.hpo_state.n_trials - self.hpo_state.completed)
                        if remaining > 0 and self._last_plan is not None:
                            nextB = min(int(self.hpo_state.batch_B), int(remaining))
                            self._start_batch(plan_template=dict(self._last_plan), batch_size=int(nextB), track=track)
                else:
                    # Non-HPO: one-shot train already issued; nothing else to do.
                    self.publish(
                        topic_type="thought",
                        kind="event",
                        payload={"event": "thought", "track": track, "summary": "Non-HPO run: evaluation received; no further actions."},
                        correlation_id=msg.correlation_id,
                    )
                    self.publish(topic_type="action", kind="event", payload={"event": "action", "track": track, "action": "idle"}, correlation_id=msg.correlation_id)

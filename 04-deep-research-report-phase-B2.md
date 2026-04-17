# Phase B Step Two Implementation for Agentic Baselines

## Design principles and message flow

This Step Two implementation turns your Phase A artifacts (processed Parquet + scalers + `dataset_manifest.json`) into a **minimal, runnable, auditable multi‑agent orchestration** that executes the baseline loop:

**Task configuration → Prepare → Ready → Model plan → Train → Evaluate**

The design choices mirror common best practices in message‑driven multi‑agent systems: agents exchange discrete **commands/events/results** through a broker/pool, and replies carry **correlation IDs** to tie messages to a run and request chain. citeturn0search0turn0search3

Your Pub/Sub implementation also follows broadcast semantics used in agent frameworks: publish to a topic; subscribed agents receive it; and by default, published messages are not delivered back to the publishing agent to avoid loops. citeturn1search9

The baseline model choices are:
- **Linear Regression** (fast, deterministic)
- **XGBoost** via `XGBRegressor` (strong tabular baseline; supports sklearn API). citeturn0search5  
And the HPO skeleton uses **Optuna** studies (`create_study`, `study.ask/tell`) to support a minimal “interactive BO” loop. citeturn1search6turn1search5

For artifact persistence, we use **joblib** (efficient for numpy heavy objects), with the standard warning that joblib/pickle should only be loaded from trusted sources. citeturn1search1turn1search0

## Directory structure and dependencies

### Expected folder layout for Phase B

Create/extend:

```
src/
  agents/
    __init__.py
    messages.py
    message_pool.py
    base_agent.py
    task_manager_agent.py
    preparation_agent.py
    model_manager_agent.py
    model_developer_agent.py
logs/
  message_pool.jsonl
  runs/
    <run_id>/
      models/
      plots/
      reports/
orchestrator.py
requirements_phase_b.txt
```

If `orchestrator.py` is at the project root, ensure:
- `src/__init__.py` exists, and
- you run with project root on `PYTHONPATH` (often already true if you run from the root).

### requirements_phase_b.txt

This file adds Phase‑B‑specific packages (your Phase A requirements already include numpy/pandas/sklearn/torch/matplotlib).

```txt
# Phase B additions
xgboost>=2.0,<4.0
optuna>=4.0,<5.0
joblib>=1.3,<2.0

# Optional (nice-to-have)
tqdm>=4.66,<5.0
```

## BaseAgent and specialized agents

All agents share a simple interface:
- `subscribe()` (once, during init)
- `tick()` (called periodically by orchestrator heartbeat)
- `publish()` helper to send Message objects

### src/agents/base_agent.py

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .message_pool import MessagePool
from .messages import Message, TopicId


@dataclass
class AgentConfig:
    agent_id: str
    role: str
    run_id: str


class BaseAgent:
    """
    Base class for research agents communicating via MessagePool.

    Each agent:
      - subscribes to topic types for a given run_id scope (topic source)
      - polls messages on heartbeat ticks
      - publishes commands/events/results
    """
    def __init__(self, cfg: AgentConfig, pool: MessagePool):
        self.agent_id = cfg.agent_id
        self.role = cfg.role
        self.run_id = cfg.run_id
        self.pool = pool

    def subscribe(self, topics: Iterable[str], *, start_from: str = "latest") -> None:
        self.pool.subscribe(
            agent_id=self.agent_id,
            agent_role=self.role,
            topics=set(topics),
            topic_source=self.run_id,
            start_from=start_from,
        )

    def poll(self, *, max_messages: int = 50) -> List[Message]:
        return self.pool.poll(self.agent_id, max_messages=max_messages)

    def publish(
        self,
        *,
        topic_type: str,
        kind: str,
        payload: dict,
        correlation_id: Optional[str] = None,
        level: str = "info",
        to: Optional[list[str]] = None,
        to_roles: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> Message:
        msg = Message.new(
            topic=TopicId(topic_type, self.run_id),
            kind=kind,  # "command" | "event" | "result" | "error"
            sender=self.agent_id,
            sender_role=self.role,
            payload=payload,
            correlation_id=correlation_id,
            run_id=self.run_id,
            level=level,
            to=to,
            to_roles=to_roles,
            tags=tags,
        )
        return self.pool.publish(msg)

    def tick(self) -> None:
        """Override in subclasses."""
        raise NotImplementedError
```

### src/agents/task_manager_agent.py

Publishes a `prepare` command once per run.

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from .base_agent import BaseAgent


@dataclass
class RunConfig:
    manifest_path: str
    track: str               # "load" or "solar"
    model_type: str          # "linear" or "xgboost"
    window: dict             # {"W": 168, "H": 24, "stride": 24}
    training: dict           # {"max_train_windows": 20000, "max_test_windows": 5000, "seed": 42}
    hpo: dict                # {"enabled": False, "n_trials": 1}


class TaskManagerAgent(BaseAgent):
    def __init__(self, cfg, pool, run_cfg: RunConfig):
        super().__init__(cfg, pool)
        self.run_cfg = run_cfg
        self._started = False
        # Task Manager can subscribe to system/evaluate if you want; not required for minimal flow.
        self.subscribe(topics=["system"], start_from="latest")

    def tick(self) -> None:
        if self._started:
            return

        self._started = True
        corr_id = str(uuid.uuid4())

        # Publish run metadata (optional)
        self.publish(
            topic_type="system",
            kind="event",
            payload={
                "event": "run_started",
                "track": self.run_cfg.track,
                "model_type": self.run_cfg.model_type,
                "window": self.run_cfg.window,
                "training": self.run_cfg.training,
                "hpo": self.run_cfg.hpo,
            },
            correlation_id=corr_id,
        )

        # Publish PREPARE command
        self.publish(
            topic_type="prepare",
            kind="command",
            payload={
                "command": "prepare",
                "manifest_path": self.run_cfg.manifest_path,
                "track": self.run_cfg.track,
            },
            correlation_id=corr_id,
        )
```

### src/agents/preparation_agent.py

Verifies Phase A artifacts and signals readiness.

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .base_agent import BaseAgent


class PreparationAgent(BaseAgent):
    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        self.subscribe(topics=["prepare"], start_from="latest")

    def _verify_manifest_and_paths(self, manifest_path: str, track: str) -> dict:
        mp = Path(manifest_path)
        if not mp.exists():
            raise FileNotFoundError(f"dataset_manifest.json not found: {mp}")

        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if "tracks" not in manifest or track not in manifest["tracks"]:
            raise ValueError(f"Track '{track}' not found in manifest['tracks'].")

        t = manifest["tracks"][track]
        paths = t["paths"]

        required = ["train_scaled", "val_scaled", "test_scaled", "scaler"]
        missing = [k for k in required if k not in paths]
        if missing:
            raise ValueError(f"Manifest missing required paths for track '{track}': {missing}")

        # Verify file existence
        for k in required:
            p = Path(paths[k])
            if not p.exists():
                raise FileNotFoundError(f"Missing artifact for '{track}': {k} -> {p}")

        # Lightweight parquet sanity check (columns exist)
        df_head = pd.read_parquet(paths["train_scaled"]).head(5)

        cols = t.get("columns", {})
        for must in ["hist_cols", "future_cols", "target_cols"]:
            if must not in cols or not cols[must]:
                raise ValueError(f"Manifest columns missing '{must}' for track '{track}'.")

        return {
            "manifest_path": str(mp),
            "track": track,
            "paths": paths,
            "columns": cols,
            "train_head_columns": list(df_head.columns),
        }

    def tick(self) -> None:
        for msg in self.poll(max_messages=20):
            if msg.kind != "command":
                continue
            if msg.payload.get("command") != "prepare":
                continue

            corr_id = msg.correlation_id
            try:
                info = self._verify_manifest_and_paths(
                    manifest_path=msg.payload["manifest_path"],
                    track=msg.payload["track"],
                )
                # Publish READY event (on prepare topic)
                self.publish(
                    topic_type="prepare",
                    kind="event",
                    payload={
                        "event": "ready",
                        "track": info["track"],
                        "manifest_path": info["manifest_path"],
                        "paths": info["paths"],
                        "columns": info["columns"],
                    },
                    correlation_id=corr_id,
                )
            except Exception as e:
                self.publish(
                    topic_type="prepare",
                    kind="error",
                    payload={
                        "event": "not_ready",
                        "error": str(e),
                    },
                    correlation_id=corr_id,
                    level="error",
                )
```

### src/agents/model_manager_agent.py

Chooses model type and emits a `train` command. Includes an Optuna ask/tell skeleton.

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import optuna  # skeleton for interactive Bayesian optimization
# Optuna create_study/optimize is standard usage; we use ask/tell for an interactive loop. citeturn1search6turn1search5

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
    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        self.subscribe(topics=["prepare", "evaluate"], start_from="latest")
        self.hpo_state: Optional[HPOState] = None
        self._last_plan: Optional[dict] = None
        self._study: Optional[optuna.Study] = None
        self._trial = None  # optuna.trial.Trial (ask/tell)

    def _make_default_params(self, model_type: str) -> Dict[str, Any]:
        if model_type == "linear":
            return {"fit_intercept": True}
        if model_type == "xgboost":
            # XGBRegressor supports sklearn API and common hyperparameters. citeturn0search5
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

        # Optional: persistent storage for study (sqlite)
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
        self.hpo_state.current_trial_number = self._trial.number

        if model_type == "xgboost":
            # Minimal search space example (expand later)
            default_params = dict(default_params)
            default_params["max_depth"] = self._trial.suggest_int("max_depth", 3, 10)
            default_params["learning_rate"] = self._trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            default_params["subsample"] = self._trial.suggest_float("subsample", 0.5, 1.0)
            default_params["colsample_bytree"] = self._trial.suggest_float("colsample_bytree", 0.5, 1.0)
            default_params["reg_lambda"] = self._trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            default_params["n_estimators"] = self._trial.suggest_int("n_estimators", 100, 800)
            return default_params

        # For linear regression, keep fixed (or tune ridge alpha later)
        return default_params

    def tick(self) -> None:
        for msg in self.poll(max_messages=50):
            # READY -> plan training
            if msg.topic.startswith(f"prepare/{self.run_id}") and msg.kind == "event" and msg.payload.get("event") == "ready":
                corr_id = msg.correlation_id
                base_cfg = msg.payload

                # The TaskManager config is not embedded by default; you can add it if desired.
                # For minimal runnable flow, we infer from environment variables or include in the READY payload later.
                # Here, we expect orchestrator to include run config in system messages; simplest: store last system config.
                # We'll accept model type from manifest-ready payload if included; otherwise defaults.
                track = base_cfg["track"]
                run_cfg = base_cfg.get("run_config", {})

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

                self._last_plan = plan

                self.publish(
                    topic_type="train",
                    kind="command",
                    payload=plan,
                    correlation_id=corr_id,
                )

            # EVALUATION result -> tell Optuna (skeleton)
            if msg.topic.startswith(f"evaluate/{self.run_id}") and msg.kind == "result":
                if self.hpo_state and self.hpo_state.enabled and self._study and self._trial is not None:
                    # Use RMSE as objective (minimize)
                    rmse = float(msg.payload.get("metrics", {}).get("rmse", float("nan")))
                    if rmse == rmse:  # not nan
                        self._study.tell(self._trial, rmse)
                        self.hpo_state.completed += 1

                        # Skeleton: stop after first trial by default
                        # Extend later: if completed < n_trials, publish another train command.
```

### src/agents/model_developer_agent.py

Loads Parquet data, builds windows, trains model, evaluates, saves `.joblib` + plot, publishes evaluation result.

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb  # XGBRegressor sklearn interface citeturn0search5

from src.data_loaders.windowing import SlidingWindowDataset, WindowConfig


from .base_agent import BaseAgent


@dataclass
class TrainOutcome:
    model_path: str
    plot_path: str
    metrics: Dict[str, float]
    n_train_windows: int
    n_test_windows: int


class ModelDeveloperAgent(BaseAgent):
    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        self.subscribe(topics=["train"], start_from="latest")

        self.run_dir = Path("logs") / "runs" / self.run_id
        (self.run_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)

    def _load_df(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _build_xy_from_windows(
        self,
        df: pd.DataFrame,
        hist_cols: list[str],
        future_cols: list[str],
        target_cols: list[str],
        W: int,
        H: int,
        stride: int,
        max_windows: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ds = SlidingWindowDataset(
            df=df,
            hist_cols=hist_cols,
            future_cols=future_cols,
            target_cols=target_cols,
            cfg=WindowConfig(W=W, H=H, stride=stride),
            return_times=False,
        )

        n = len(ds)
        take = min(n, max_windows) if max_windows > 0 else n

        X_list = []
        Y_list = []

        for i in range(take):
            x_hist_t, x_fut_t, y_t = ds[i]
            x_hist = x_hist_t.numpy()
            x_fut = x_fut_t.numpy()
            y = y_t.numpy()

            x_vec = np.concatenate([x_hist.reshape(-1), x_fut.reshape(-1)], axis=0)
            y_vec = y.reshape(-1)  # H * D_y

            X_list.append(x_vec)
            Y_list.append(y_vec)

        X = np.stack(X_list, axis=0).astype(np.float32)
        Y = np.stack(Y_list, axis=0).astype(np.float32)
        return X, Y

    def _make_model(self, model_type: str, hyperparams: dict):
        if model_type == "linear":
            # LinearRegression supports multi-output natively when y is 2D
            return LinearRegression(**{k: v for k, v in hyperparams.items() if k in ["fit_intercept"]})

        if model_type == "xgboost":
            base = xgb.XGBRegressor(
                objective="reg:squarederror",
                **hyperparams
            )
            # Multi-output wrapper for horizon>1
            return MultiOutputRegressor(base)

        raise ValueError(f"Unsupported model_type: {model_type}")

    def _inverse_targets_if_possible(self, scaler_obj: Any, y_scaled: np.ndarray) -> np.ndarray:
        # Your LeakageSafeScaler implements inverse_transform_targets(y_scaled)
        if hasattr(scaler_obj, "inverse_transform_targets"):
            return scaler_obj.inverse_transform_targets(y_scaled)
        return y_scaled

    def _evaluate_and_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plot_path: Path,
        horizon: int,
        max_points: int = 400,
    ) -> Dict[str, float]:
        # Aggregate metrics over all horizons (flatten)
        mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))
        rmse = float(math.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))))

        # Plot step-1 horizon only (column 0) for readability
        y_t0 = y_true[:, 0]
        y_p0 = y_pred[:, 0]

        n = min(len(y_t0), max_points)
        y_t0 = y_t0[:n]
        y_p0 = y_p0[:n]

        plt.figure()
        plt.plot(y_t0, label="Actual")
        plt.plot(y_p0, label="Predicted")
        plt.title("Predicted vs Actual (Horizon step 1)")
        plt.xlabel("Window index")
        plt.ylabel("Target")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path.as_posix(), dpi=150)
        plt.close()

        return {"mae": mae, "rmse": rmse}

    def _train_once(self, payload: dict) -> TrainOutcome:
        track = payload["track"]
        paths = payload["paths"]
        cols = payload["columns"]
        model_type = payload["model_type"]
        hyperparams = payload["hyperparams"]

        W = int(payload["window"]["W"])
        H = int(payload["window"]["H"])
        stride = int(payload["window"]["stride"])

        max_train = int(payload["training"].get("max_train_windows", 20000))
        max_test = int(payload["training"].get("max_test_windows", 5000))

        df_train = self._load_df(paths["train_scaled"])
        df_test = self._load_df(paths["test_scaled"])

        # Load scaler to inverse-transform metrics/plots
        scaler_obj = joblib.load(paths["scaler"])

        X_train, y_train = self._build_xy_from_windows(
            df_train,
            hist_cols=cols["hist_cols"],
            future_cols=cols["future_cols"],
            target_cols=cols["target_cols"],
            W=W, H=H, stride=stride,
            max_windows=max_train,
        )
        X_test, y_test = self._build_xy_from_windows(
            df_test,
            hist_cols=cols["hist_cols"],
            future_cols=cols["future_cols"],
            target_cols=cols["target_cols"],
            W=W, H=H, stride=stride,
            max_windows=max_test,
        )

        model = self._make_model(model_type, hyperparams)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Inverse-transform for metrics in original scale
        y_test_inv = self._inverse_targets_if_possible(scaler_obj, y_test)
        y_pred_inv = self._inverse_targets_if_possible(scaler_obj, y_pred)

        model_path = self.run_dir / "models" / f"{track}_{model_type}.joblib"
        joblib.dump(model, model_path)

        plot_path = self.run_dir / "plots" / f"{track}_{model_type}_pred_vs_actual.png"
        metrics = self._evaluate_and_plot(
            y_true=y_test_inv,
            y_pred=y_pred_inv,
            plot_path=plot_path,
            horizon=H,
            max_points=400,
        )

        return TrainOutcome(
            model_path=model_path.as_posix(),
            plot_path=plot_path.as_posix(),
            metrics=metrics,
            n_train_windows=int(len(X_train)),
            n_test_windows=int(len(X_test)),
        )

    def tick(self) -> None:
        for msg in self.poll(max_messages=10):
            if msg.kind != "command":
                continue
            if msg.payload.get("command") != "train":
                continue

            corr_id = msg.correlation_id
            try:
                outcome = self._train_once(msg.payload)

                self.publish(
                    topic_type="evaluate",
                    kind="result",
                    payload={
                        "event": "evaluation_complete",
                        "track": msg.payload["track"],
                        "model_type": msg.payload["model_type"],
                        "metrics": outcome.metrics,
                        "artifacts": {
                            "model_path": outcome.model_path,
                            "plot_path": outcome.plot_path,
                        },
                        "data_used": {
                            "n_train_windows": outcome.n_train_windows,
                            "n_test_windows": outcome.n_test_windows,
                            "window": msg.payload["window"],
                        },
                        "optuna": msg.payload.get("optuna", {"enabled": False}),
                    },
                    correlation_id=corr_id,
                )

            except Exception as e:
                self.publish(
                    topic_type="evaluate",
                    kind="error",
                    payload={
                        "event": "evaluation_failed",
                        "error": str(e),
                        "track": msg.payload.get("track"),
                        "model_type": msg.payload.get("model_type"),
                    },
                    correlation_id=corr_id,
                    level="error",
                )
```

## Orchestrator entry point

### orchestrator.py (project root)

This runs a heartbeat loop. Each second it calls `tick()` on all agents and stops once it sees an evaluation result for this run.

```python
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

from src.agents.message_pool import MessagePool
from src.agents.base_agent import AgentConfig
from src.agents.task_manager_agent import TaskManagerAgent, RunConfig
from src.agents.preparation_agent import PreparationAgent
from src.agents.model_manager_agent import ModelManagerAgent
from src.agents.model_developer_agent import ModelDeveloperAgent


def make_run_id() -> str:
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


def parse_args():
    p = argparse.ArgumentParser(description="Phase B Orchestrator: agentic baseline training loop")
    p.add_argument("--manifest", type=str, default="data/processed/dataset_manifest.json")
    p.add_argument("--track", type=str, choices=["load", "solar"], default="load")
    p.add_argument("--model", type=str, choices=["linear", "xgboost"], default="xgboost")

    p.add_argument("--W", type=int, default=168)
    p.add_argument("--H", type=int, default=24)
    p.add_argument("--stride", type=int, default=24)

    p.add_argument("--max_train_windows", type=int, default=20000)
    p.add_argument("--max_test_windows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hpo", action="store_true", help="Enable Optuna skeleton (1 trial by default)")
    p.add_argument("--n_trials", type=int, default=1)

    p.add_argument("--tick_seconds", type=float, default=1.0)
    p.add_argument("--timeout_seconds", type=float, default=600.0)
    return p.parse_args()


def main():
    args = parse_args()
    run_id = make_run_id()

    pool = MessagePool(log_path="logs/message_pool.jsonl", load_existing=True)

    run_cfg = RunConfig(
        manifest_path=args.manifest,
        track=args.track,
        model_type=args.model,
        window={"W": args.W, "H": args.H, "stride": args.stride},
        training={"max_train_windows": args.max_train_windows, "max_test_windows": args.max_test_windows, "seed": args.seed},
        hpo={"enabled": bool(args.hpo), "n_trials": int(args.n_trials), "study_name": f"study_{run_id}", "storage": None},
    )

    # Agents
    task_manager = TaskManagerAgent(
        AgentConfig(agent_id="TaskManagerAgent", role="task_manager", run_id=run_id),
        pool,
        run_cfg,
    )
    prep = PreparationAgent(
        AgentConfig(agent_id="PreparationAgent", role="preparation", run_id=run_id),
        pool,
    )
    model_manager = ModelManagerAgent(
        AgentConfig(agent_id="ModelManagerAgent", role="model_manager", run_id=run_id),
        pool,
    )
    model_dev = ModelDeveloperAgent(
        AgentConfig(agent_id="ModelDeveloperAgent", role="model_developer", run_id=run_id),
        pool,
    )

    agents = [task_manager, prep, model_manager, model_dev]

    start = time.time()
    done = False

    while not done and (time.time() - start) < args.timeout_seconds:
        for a in agents:
            a.tick()

        # Stop when evaluation result exists
        eval_msgs = pool.find_by_topic("evaluate", topic_source=run_id)
        for m in eval_msgs:
            if m.kind == "result" and m.payload.get("event") == "evaluation_complete":
                done = True
                break
            if m.kind == "error":
                done = True
                break

        time.sleep(args.tick_seconds)

    if not done:
        print(f"[WARN] Timeout reached without completion. run_id={run_id}")
    else:
        print(f"[OK] Run completed. run_id={run_id}")
        print("See logs/message_pool.jsonl and logs/runs/<run_id>/ for artifacts.")


if __name__ == "__main__":
    main()
```

## How the baseline uses the SlidingWindowDataset

Tabular baselines (Linear Regression, XGBoost) require a 2D feature matrix **X**. Your window dataset provides:
- `X_hist`: `[W, D_hist]`
- `X_future`: `[H, D_future]`
- `Y`: `[H, D_y]`

This implementation creates a **single vector per window**:

\[
X = \text{flatten}(X_{hist}) \; \Vert \; \text{flatten}(X_{future}), \quad
y = \text{flatten}(Y)
\]

and trains a multi-output regressor for horizon \(H\) (XGBoost uses `MultiOutputRegressor` wrapping `XGBRegressor`, which is supported as an sklearn regressor). citeturn0search5

This is a clean baseline because it:
- respects the time-window structure,
- produces a full multi-horizon output,
- and remains simple enough to validate the agentic orchestration before adding transformers.

## What you should see in `logs/message_pool.jsonl`

Once you run:

```bash
python orchestrator.py --track load --model xgboost
```

you should see messages (one per JSONL line) progressing like:

- `system/run_id` — `run_started` event  
- `prepare/run_id` — `prepare` command  
- `prepare/run_id` — `ready` event (manifest verified)  
- `train/run_id` — `train` command (model plan + hyperparams + window config)  
- `evaluate/run_id` — `evaluation_complete` result (MAE, RMSE + artifact paths)  

This gives you the “research auditability” property you want: append‑only JSONL logs can be replayed, filtered by `run_id`, and correlated by `correlation_id`, consistent with message-driven multi-agent best practices. citeturn0search0turn1search9
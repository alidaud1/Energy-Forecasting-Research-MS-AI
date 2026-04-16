from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb

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
    """
    Trains a baseline model from scaled parquet splits, evaluates, saves artifacts,
    and publishes an evaluation result.
    """

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
        *,
        require_finite_targets: bool = True,
        progress_label: str = "",
        correlation_id: str | None = None,
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
        skipped = 0

        for i in range(take):
            x_hist_t, x_fut_t, y_t = ds[i]
            x_hist = x_hist_t.numpy()
            x_fut = x_fut_t.numpy()
            y = y_t.numpy()

            if require_finite_targets and not np.isfinite(y).all():
                # Solar has POWER missing in the forecast month; skip any window whose targets are not finite.
                skipped += 1
                continue

            x_vec = np.concatenate([x_hist.reshape(-1), x_fut.reshape(-1)], axis=0)
            y_vec = y.reshape(-1)

            X_list.append(x_vec)
            Y_list.append(y_vec)

            if correlation_id and (i + 1) % 500 == 0:
                self.publish(
                    topic_type="train",
                    kind="event",
                    payload={
                        "event": "window_build_progress",
                        "stage": progress_label,
                        "seen": int(i + 1),
                        "kept": int(len(X_list)),
                        "skipped": int(skipped),
                        "max_windows": int(take),
                    },
                    correlation_id=correlation_id,
                )

        if not X_list:
            raise ValueError("No usable windows found (targets may be missing/NaN for chosen split/window config).")

        X = np.stack(X_list, axis=0).astype(np.float32)
        Y = np.stack(Y_list, axis=0).astype(np.float32)
        return X, Y

    def _make_model(self, model_type: str, hyperparams: dict):
        if model_type == "linear":
            return LinearRegression(**{k: v for k, v in hyperparams.items() if k in ["fit_intercept"]})

        if model_type == "xgboost":
            base = xgb.XGBRegressor(objective="reg:squarederror", **hyperparams)
            return MultiOutputRegressor(base)

        raise ValueError(f"Unsupported model_type: {model_type}")

    def _inverse_targets_if_possible(self, scaler_obj: Any, y_scaled: np.ndarray) -> np.ndarray:
        """
        The dataframe target scaling is per-timepoint (1 column target). Windowing flattens (H * D_y)
        into a wide vector, so we must reshape before inverse-transforming.
        """
        if not hasattr(scaler_obj, "inverse_transform_targets"):
            return y_scaled

        target_cols = getattr(scaler_obj, "target_cols", None) or ["y"]
        n_targets = int(len(target_cols))

        if y_scaled.ndim != 2:
            return y_scaled

        n = y_scaled.shape[0]
        d = y_scaled.shape[1]
        if d == n_targets:
            return scaler_obj.inverse_transform_targets(y_scaled)

        if d % n_targets != 0:
            return y_scaled

        y2 = y_scaled.reshape(-1, n_targets)
        y2_inv = scaler_obj.inverse_transform_targets(y2)
        return y2_inv.reshape(n, d)

    def _evaluate_and_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plot_path: Path,
        max_points: int = 400,
    ) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))
        rmse = float(math.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))))

        y_t0 = y_true[:, 0]
        y_p0 = y_pred[:, 0]

        n = min(len(y_t0), max_points)
        plt.figure()
        plt.plot(y_t0[:n], label="Actual")
        plt.plot(y_p0[:n], label="Predicted")
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

        scaler_obj = joblib.load(paths["scaler"])

        corr_id = payload.get("_correlation_id")
        self.publish(
            topic_type="train",
            kind="event",
            payload={
                "event": "training_started",
                "track": track,
                "model_type": model_type,
                "window": {"W": W, "H": H, "stride": stride},
                "limits": {"max_train_windows": max_train, "max_test_windows": max_test},
            },
            correlation_id=corr_id,
        )

        X_train, y_train = self._build_xy_from_windows(
            df_train,
            hist_cols=cols["hist_cols"],
            future_cols=cols["future_cols"],
            target_cols=cols["target_cols"],
            W=W,
            H=H,
            stride=stride,
            max_windows=max_train,
            require_finite_targets=True,
            progress_label="train",
            correlation_id=corr_id,
        )
        X_test, y_test = self._build_xy_from_windows(
            df_test,
            hist_cols=cols["hist_cols"],
            future_cols=cols["future_cols"],
            target_cols=cols["target_cols"],
            W=W,
            H=H,
            stride=stride,
            max_windows=max_test,
            require_finite_targets=True,
            progress_label="test",
            correlation_id=corr_id,
        )

        model = self._make_model(model_type, hyperparams)
        self.publish(
            topic_type="train",
            kind="event",
            payload={"event": "model_fit_started", "track": track, "model_type": model_type, "x_shape": list(X_train.shape)},
            correlation_id=corr_id,
        )
        model.fit(X_train, y_train)
        self.publish(
            topic_type="train",
            kind="event",
            payload={"event": "model_fit_complete", "track": track, "model_type": model_type},
            correlation_id=corr_id,
        )

        y_pred = model.predict(X_test)

        y_test_inv = self._inverse_targets_if_possible(scaler_obj, y_test)
        y_pred_inv = self._inverse_targets_if_possible(scaler_obj, y_pred)

        model_path = self.run_dir / "models" / f"{track}_{model_type}.joblib"
        joblib.dump(model, model_path)

        plot_path = self.run_dir / "plots" / f"{track}_{model_type}_pred_vs_actual.png"
        metrics = self._evaluate_and_plot(y_true=y_test_inv, y_pred=y_pred_inv, plot_path=plot_path)

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
                payload = dict(msg.payload)
                payload["_correlation_id"] = corr_id
                outcome = self._train_once(payload)
                self.publish(
                    topic_type="evaluate",
                    kind="result",
                    payload={
                        "event": "evaluation_complete",
                        "track": msg.payload.get("track"),
                        "model_type": msg.payload.get("model_type"),
                        "metrics": outcome.metrics,
                        "artifacts": {"model_path": outcome.model_path, "plot_path": outcome.plot_path},
                        "n_train_windows": outcome.n_train_windows,
                        "n_test_windows": outcome.n_test_windows,
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

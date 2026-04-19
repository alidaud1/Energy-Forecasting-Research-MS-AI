from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from src.models.transformers import (
    EarlyStoppingConfig,
    ITransformerConfig,
    ITransformerForecaster,
    PatchTransformerConfig,
    PatchTransformerForecaster,
    TrainerConfig,
    predict_batches,
    train_model,
)

import torch
from torch.utils.data import DataLoader, Dataset

from .base_agent import BaseAgent


@dataclass
class TrainOutcome:
    model_path: str
    plot_path: str
    predictions_path: str
    metrics: Dict[str, float]
    n_train_windows: int
    n_test_windows: int


@dataclass
class TransformerExecutionConfig:
    model_name: str  # "patchtst" | "itransformer"
    strategy: str  # "A" | "B"
    add_future_indicator: bool
    use_hour_embedding: bool


class TransformerInputDataset(Dataset):
    """
    Wrap SlidingWindowDataset to produce transformer-ready (X_in, Y) pairs.

    Base ds returns:
      X_hist:   [W, D_hist]
      X_future: [H, D_future]
      Y:        [H, D_y]

    Output:
      Strategy A: X_in = X_hist                      -> [W, D_hist]
      Strategy B: X_in = concat history + future cov -> [W+H, D_hist] (+ optional indicator channel)
    Optionally emits hour_idx: [L] for PatchTransformer hour embeddings.
    """

    def __init__(
        self,
        base_ds: SlidingWindowDataset,
        hist_cols: List[str],
        future_cols: List[str],
        target_cols: List[str],
        *,
        strategy: str,
        add_future_indicator: bool,
        include_hour_idx: bool,
        max_windows: int = 0,
        require_finite_targets: bool = True,
    ):
        self.ds = base_ds
        self.hist_cols = hist_cols
        self.future_cols = future_cols
        self.target_cols = target_cols
        self.strategy = strategy.upper()
        self.add_future_indicator = add_future_indicator
        self.include_hour_idx = include_hour_idx
        self.require_finite_targets = require_finite_targets

        self.W = base_ds.cfg.W
        self.H = base_ds.cfg.H

        hist_index = {c: i for i, c in enumerate(hist_cols)}
        self.future_to_hist = {j: hist_index[c] for j, c in enumerate(future_cols) if c in hist_index}

        # Build a filtered index list (mainly to handle solar missing targets in test/clean).
        idxs: List[int] = []
        n = len(base_ds)
        take = min(n, max_windows) if max_windows and max_windows > 0 else n
        for i in range(take):
            _xh, _xf, y = base_ds[i]
            if require_finite_targets and not torch.isfinite(y).all():
                continue
            idxs.append(i)
        self._idxs = idxs
        if not self._idxs:
            raise ValueError("No usable windows found for transformer dataset (targets may be NaN for chosen split/window config).")

    def __len__(self) -> int:
        return len(self._idxs)

    @property
    def base_indices(self) -> List[int]:
        # Maps dataset index -> base SlidingWindowDataset index.
        return list(self._idxs)

    def _make_hour_idx(self, base_idx: int, L: int) -> torch.Tensor:
        _zone, _hs, he, _fs, _fe = self.ds.window_time_bounds(base_idx)
        hist_start = he - pd.Timedelta(hours=self.W - 1)
        times = pd.date_range(hist_start, periods=L, freq="h")
        return torch.tensor(times.hour.to_numpy(dtype=np.int64), dtype=torch.long)

    def __getitem__(self, idx: int):
        base_idx = self._idxs[idx]
        x_hist, x_fut, y = self.ds[base_idx]
        W, Dh = x_hist.shape
        H, Df = x_fut.shape

        if self.strategy == "A":
            x_in = x_hist
        else:
            x_full = torch.zeros((W + H, Dh), dtype=x_hist.dtype)
            x_full[:W, :] = x_hist
            for j in range(Df):
                if j in self.future_to_hist:
                    x_full[W:, self.future_to_hist[j]] = x_fut[:, j]
            if self.add_future_indicator:
                ind = torch.zeros((W + H, 1), dtype=x_hist.dtype)
                ind[W:, 0] = 1.0
                x_full = torch.cat([x_full, ind], dim=1)
            x_in = x_full

        if self.include_hour_idx:
            hour_idx = self._make_hour_idx(base_idx, x_in.shape[0])
            return x_in, y, hour_idx
        return x_in, y


class ModelDeveloperAgent(BaseAgent):
    """
    Trains a baseline model from scaled parquet splits, evaluates, saves artifacts,
    and publishes an evaluation result.
    """

    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        self.subscribe(topics=["train", "train_execute"], start_from="latest")

        self.run_dir = Path("logs") / "runs" / self.run_id
        (self.run_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "child_runs").mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_df(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _out_dir(self, child_run_id: Optional[str]) -> Path:
        if not child_run_id:
            return self.run_dir
        out = self.run_dir / "child_runs" / child_run_id
        (out / "models").mkdir(parents=True, exist_ok=True)
        (out / "plots").mkdir(parents=True, exist_ok=True)
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        return out

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
        return_base_indices: bool = False,
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

        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        kept_base_idxs: List[int] = []
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
            kept_base_idxs.append(i)

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
        if return_base_indices:
            return X, Y, kept_base_idxs  # type: ignore[return-value]
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
        child_run_id = str(payload.get("child_run_id") or "")
        parent_run_id = str(payload.get("parent_run_id") or "")
        comparison_group_id = str(payload.get("comparison_group_id") or "")
        variant = str(payload.get("variant") or "proposed")
        out_dir = self._out_dir(child_run_id or None)

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
        # Also keep base window indices so we can reconstruct forecast timestamps in predictions parquet.
        X_test, y_test, test_base_idxs = self._build_xy_from_windows(
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
            return_base_indices=True,
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

        model_path = out_dir / "models" / f"{track}_{model_type}.joblib"
        joblib.dump(model, model_path)

        plot_path = out_dir / "plots" / f"{track}_{model_type}_pred_vs_actual.png"
        metrics = self._evaluate_and_plot(y_true=y_test_inv, y_pred=y_pred_inv, plot_path=plot_path)

        # Write baseline predictions parquet for Phase C3 integration/plots.
        dy = len(cols.get("target_cols") or ["y"])
        if dy != 1:
            raise ValueError("baseline predictions parquet currently supports single-target only")
        if y_test_inv.shape[1] != H * dy:
            raise ValueError("unexpected baseline y_test shape for parquet export")
        y_true_3d = y_test_inv.reshape(-1, H, dy)
        y_pred_3d = y_pred_inv.reshape(-1, H, dy)
        ds_test_base = SlidingWindowDataset(
            df=df_test,
            hist_cols=cols["hist_cols"],
            future_cols=cols["future_cols"],
            target_cols=cols["target_cols"],
            cfg=WindowConfig(W=W, H=H, stride=stride),
            return_times=False,
        )
        pred_path = out_dir / "predictions" / f"{track}_{model_type}_baseline_predictions.parquet"
        pred_path_str = self._save_predictions_parquet(
            ds_test_base,
            y_true_3d,
            y_pred_3d,
            pred_path,
            track=track,
            model_family="baseline",
            model_name=model_type,
            strategy="baseline",
            base_indices=test_base_idxs,
            run_id_override=child_run_id or None,
            parent_run_id=parent_run_id or None,
            variant=variant,
            comparison_group_id=comparison_group_id,
        )

        return TrainOutcome(
            model_path=model_path.as_posix(),
            plot_path=plot_path.as_posix(),
            predictions_path=pred_path_str,
            metrics=metrics,
            n_train_windows=int(len(X_train)),
            n_test_windows=int(len(X_test)),
        )

    def _inverse_targets_3d(self, scaler_obj: Any, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse-transform transformer outputs in shape [N, H, Dy] by reshaping to 2D.
        """
        if not hasattr(scaler_obj, "inverse_transform_targets"):
            return y_scaled
        if y_scaled.ndim != 3:
            return y_scaled
        N, H, Dy = y_scaled.shape
        y2 = y_scaled.reshape(N * H, Dy)
        y2_inv = scaler_obj.inverse_transform_targets(y2)
        return y2_inv.reshape(N, H, Dy)

    def _save_predictions_parquet(
        self,
        base_ds: SlidingWindowDataset,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        out_path: Path,
        *,
        track: str,
        model_family: str,
        model_name: str,
        strategy: str = "",
        base_indices: Optional[List[int]] = None,
        run_id_override: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        variant: str = "raw",
        comparison_group_id: str = "",
    ) -> str:
        rows = []
        N, H, Dy = y_true.shape
        if Dy != 1:
            raise ValueError("predictions parquet writer currently supports single-target only")

        for i in range(N):
            base_i = base_indices[i] if base_indices is not None else i
            zone, _hs, _he, fs, _fe = base_ds.window_time_bounds(base_i)
            times = pd.date_range(fs, periods=H, freq="h")
            for h in range(H):
                rows.append(
                    {
                        "run_id": run_id_override or self.run_id,
                        "parent_run_id": parent_run_id or "",
                        "variant": variant,
                        "comparison_group_id": comparison_group_id or "",
                        "track": track,
                        "model_family": model_family,
                        "model_name": model_name,
                        "ZONEID": int(zone),
                        "timestamp": times[h],
                        "horizon": int(h + 1),
                        "y_true": float(y_true[i, h, 0]),
                        "y_pred": float(y_pred[i, h, 0]),
                        # Backward/diagnostic fields (safe to keep)
                        "model": model_name,
                        "strategy": strategy,
                    }
                )

        dfp = pd.DataFrame(rows)
        dfp.to_parquet(out_path)
        return out_path.as_posix()

    def _build_transformer_model(self, model_name: str, *, input_dim: int, context_length: int, pred_length: int, use_hour_embedding: bool) -> torch.nn.Module:
        model_name = model_name.lower()
        if model_name == "patchtst":
            cfg = PatchTransformerConfig(
                input_dim=input_dim,
                context_length=context_length,
                pred_length=pred_length,
                target_dim=1,
                patch_length=16,
                patch_stride=16,
                d_model=256,
                n_heads=8,
                n_layers=4,
                d_ff=512,
                dropout=0.1,
                use_cls_token=True,
                use_hour_embedding=use_hour_embedding,
                hour_emb_dim=256,
            )
            return PatchTransformerForecaster(cfg)
        if model_name == "itransformer":
            cfg = ITransformerConfig(
                input_dim=input_dim,
                context_length=context_length,
                pred_length=pred_length,
                target_dim=1,
                d_model=256,
                n_heads=8,
                n_layers=4,
                d_ff=512,
                dropout=0.1,
            )
            return ITransformerForecaster(cfg)
        raise ValueError(f"Unknown transformer model_name: {model_name}")

    def _train_transformer_once(self, payload: dict) -> Dict[str, Any]:
        track = payload["track"]
        paths = payload["paths"]
        cols = payload["columns"]
        window = payload["window"]
        training = payload.get("training", {}) or {}
        child_run_id = str(payload.get("child_run_id") or "")
        parent_run_id = str(payload.get("parent_run_id") or "")
        comparison_group_id = str(payload.get("comparison_group_id") or "")
        variant = str(payload.get("variant") or "proposed")
        out_dir = self._out_dir(child_run_id or None)
        exec_cfg_dict = payload["exec_cfg"]
        exec_cfg = TransformerExecutionConfig(
            model_name=str(exec_cfg_dict["model_name"]),
            strategy=str(exec_cfg_dict["strategy"]),
            add_future_indicator=bool(exec_cfg_dict.get("add_future_indicator", True)),
            use_hour_embedding=bool(exec_cfg_dict.get("use_hour_embedding", False)),
        )

        W = int(window["W"])
        H = int(window["H"])
        stride = int(window.get("stride", H))

        batch_size = int(training.get("batch_size", 128))
        epochs = int(training.get("epochs", 50))
        lr = float(training.get("lr", 1e-3))
        patience = int(training.get("patience", 10))

        max_train_windows = int(training.get("max_train_windows", 0))
        max_val_windows = int(training.get("max_val_windows", 0))
        max_test_windows = int(training.get("max_test_windows", 0))

        df_train = pd.read_parquet(paths["train_scaled"])
        df_val = pd.read_parquet(paths["val_scaled"])
        df_test = pd.read_parquet(paths["test_scaled"])
        scaler_obj = joblib.load(paths["scaler"])

        base_cfg = WindowConfig(W=W, H=H, stride=stride)
        ds_train_base = SlidingWindowDataset(df_train, cols["hist_cols"], cols["future_cols"], cols["target_cols"], base_cfg, return_times=False)
        ds_val_base = SlidingWindowDataset(df_val, cols["hist_cols"], cols["future_cols"], cols["target_cols"], base_cfg, return_times=False)
        ds_test_base = SlidingWindowDataset(df_test, cols["hist_cols"], cols["future_cols"], cols["target_cols"], base_cfg, return_times=False)

        ds_train = TransformerInputDataset(
            ds_train_base,
            cols["hist_cols"],
            cols["future_cols"],
            cols["target_cols"],
            strategy=exec_cfg.strategy,
            add_future_indicator=exec_cfg.add_future_indicator,
            include_hour_idx=exec_cfg.use_hour_embedding and exec_cfg.model_name.lower() == "patchtst",
            max_windows=max_train_windows,
            require_finite_targets=True,
        )
        ds_val = TransformerInputDataset(
            ds_val_base,
            cols["hist_cols"],
            cols["future_cols"],
            cols["target_cols"],
            strategy=exec_cfg.strategy,
            add_future_indicator=exec_cfg.add_future_indicator,
            include_hour_idx=exec_cfg.use_hour_embedding and exec_cfg.model_name.lower() == "patchtst",
            max_windows=max_val_windows,
            require_finite_targets=True,
        )
        ds_test = TransformerInputDataset(
            ds_test_base,
            cols["hist_cols"],
            cols["future_cols"],
            cols["target_cols"],
            strategy=exec_cfg.strategy,
            add_future_indicator=exec_cfg.add_future_indicator,
            include_hour_idx=exec_cfg.use_hour_embedding and exec_cfg.model_name.lower() == "patchtst",
            max_windows=max_test_windows,
            require_finite_targets=True,
        )

        sample = ds_train[0]
        if exec_cfg.use_hour_embedding and exec_cfg.model_name.lower() == "patchtst":
            x0, y0, _h0 = sample
        else:
            x0, y0 = sample
        context_length = int(x0.shape[0])
        input_dim = int(x0.shape[1])

        model = self._build_transformer_model(
            exec_cfg.model_name,
            input_dim=input_dim,
            context_length=context_length,
            pred_length=H,
            use_hour_embedding=exec_cfg.use_hour_embedding and exec_cfg.model_name.lower() == "patchtst",
        )

        def make_loader(ds, shuffle: bool):
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
            )

        train_loader = make_loader(ds_train, shuffle=True)
        val_loader = make_loader(ds_val, shuffle=False)
        test_loader = make_loader(ds_test, shuffle=False)

        tcfg = TrainerConfig(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=1e-4,
            grad_clip_norm=1.0,
            early_stopping=EarlyStoppingConfig(patience=patience, min_delta=0.0),
        )

        train_stats = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            cfg=tcfg,
            use_hour_embedding=exec_cfg.use_hour_embedding and exec_cfg.model_name.lower() == "patchtst",
        )

        y_pred_t, y_true_t = predict_batches(
            model,
            test_loader,
            self.device,
            use_hour_embedding=exec_cfg.use_hour_embedding and exec_cfg.model_name.lower() == "patchtst",
        )
        y_pred = y_pred_t.numpy()
        y_true = y_true_t.numpy()

        y_pred_inv = self._inverse_targets_3d(scaler_obj, y_pred)
        y_true_inv = self._inverse_targets_3d(scaler_obj, y_true)

        model_path = out_dir / "models" / f"{track}_{exec_cfg.model_name}_{exec_cfg.strategy}.pt"
        torch.save({"state_dict": model.state_dict()}, model_path)

        plot_path = out_dir / "plots" / f"{track}_{exec_cfg.model_name}_{exec_cfg.strategy}_pred_vs_actual.png"
        metrics = self._metrics_and_plot_3d(
            y_true=y_true_inv,
            y_pred=y_pred_inv,
            plot_path=plot_path,
            title=f"{track.upper()} | {exec_cfg.model_name} | Strategy {exec_cfg.strategy}",
        )

        pred_path = out_dir / "predictions" / f"{track}_{exec_cfg.model_name}_{exec_cfg.strategy}_predictions.parquet"
        pred_path_str = self._save_predictions_parquet(
            ds_test_base,
            y_true_inv,
            y_pred_inv,
            pred_path,
            track=track,
            model_family="transformer",
            model_name=exec_cfg.model_name,
            strategy=exec_cfg.strategy,
            base_indices=ds_test.base_indices,
            run_id_override=child_run_id or None,
            parent_run_id=parent_run_id or None,
            variant=variant,
            comparison_group_id=comparison_group_id,
        )

        return {
            "track": track,
            "model_family": "transformer",
            "model_name": exec_cfg.model_name,
            "strategy": exec_cfg.strategy,
            "metrics": metrics,
            "train_stats": train_stats,
            "artifacts": {"model_path": model_path.as_posix(), "plot_path": plot_path.as_posix(), "predictions_path": pred_path_str},
            "device": str(self.device),
            "child_run_id": child_run_id,
            "parent_run_id": parent_run_id,
            "comparison_group_id": comparison_group_id,
            "variant": variant,
        }

    def _metrics_and_plot_3d(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plot_path: Path,
        title: str,
        max_points: int = 400,
    ) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))
        rmse = float(math.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))))

        y_t0 = y_true[:, 0, 0]
        y_p0 = y_pred[:, 0, 0]
        n = min(len(y_t0), max_points)
        plt.figure()
        plt.plot(y_t0[:n], label="Actual")
        plt.plot(y_p0[:n], label="Predicted")
        plt.title(f"{title} | Pred vs Actual (H=1)")
        plt.xlabel("Window index")
        plt.ylabel("Target (original units)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path.as_posix(), dpi=150)
        plt.close()

        return {"mae": mae, "rmse": rmse}

    def tick(self) -> None:
        for msg in self.poll(max_messages=10):
            if msg.kind != "command":
                continue

            corr_id = msg.correlation_id
            cmd = msg.payload.get("command")

            # Phase B baseline path
            if cmd == "train":
                model_type = msg.payload.get("model_type")
                # In Phase C research mode, `train` is a high-level plan message that is
                # converted into `train_execute` by specialized forecasting agents.
                # Only baseline model types should be executed directly here.
                if model_type not in ("linear", "xgboost"):
                    continue
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
                            "model_name": msg.payload.get("model_type"),
                            "model_family": "baseline",
                            "child_run_id": msg.payload.get("child_run_id", ""),
                            "parent_run_id": msg.payload.get("parent_run_id", ""),
                            "comparison_group_id": msg.payload.get("comparison_group_id", ""),
                            "variant": msg.payload.get("variant", ""),
                            "metrics": outcome.metrics,
                            "artifacts": {"model_path": outcome.model_path, "plot_path": outcome.plot_path, "predictions_path": outcome.predictions_path},
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
                continue

            # Phase C transformer execution path
            if cmd == "train_execute":
                try:
                    result = self._train_transformer_once(msg.payload)
                    self.publish(topic_type="evaluate", kind="result", payload={"event": "evaluation_complete", **result}, correlation_id=corr_id)
                except Exception as e:
                    self.publish(
                        topic_type="evaluate",
                        kind="error",
                        payload={"event": "evaluation_failed", "error": str(e), "track": msg.payload.get("track")},
                        correlation_id=corr_id,
                        level="error",
                    )

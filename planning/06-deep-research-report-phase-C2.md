# Phase C Step One Implementation Blueprint for Transformer Training and Specialized Forecasting Agents

## Executive summary

You’re now transitioning from “flattened-window tabular baselines” to **true sequence models** where the model consumes **3D tensors** shaped **`(batch_size, sequence_length, num_input_channels)`**, which matches PatchTST’s documented input contract for `past_values`. citeturn0search1turn0search4  

This Phase C Step One deliverable focuses on:

- A **PyTorch transformer forecaster core** with a PatchTST-style *patch tokenization* (local segments become tokens) citeturn2search2turn2search4 and an iTransformer-style *variate-token inversion* (features/variates are tokens). citeturn1search0turn1search1  
- A training loop with **GPU support**, **early stopping**, and **inverse scaling** for reporting MAE/RMSE in original units.
- Specialized agents: **LoadForecastingAgent**, **SolarForecastingAgent**, and **ForecastIntegrationAgent**.
- An updated orchestrator that can run **parallel tracks** in `--mode research` and produce a **joint decision-ready report**.

Also: because PyTorch `DataLoader` automatic batching prepends a batch dimension and collates tensors into batched tensors by default, your Phase A window datasets can be used directly for `(B, W, D)` training. citeturn2search1turn2search5  

## Phase C architecture and message flow

### Recommended flow for parallel load + solar transformers

This preserves your base paper’s “agentic AI orchestrates the workflow” separation:

- **TaskManagerAgent** publishes *prepare* commands for **both** tracks.
- **PreparationAgent** verifies Phase A artifacts for both and emits *ready* events.
- **ModelManagerAgent** emits *train* commands (directed to load vs solar roles).
- **LoadForecastingAgent / SolarForecastingAgent** specialize configuration (solar: predictor selection, decumulation checks, known-future covariates strategy) and hand off an execution request.
- **ModelDeveloperAgent** runs the actual training loop, saves artifacts, and publishes evaluation events.
- **ForecastIntegrationAgent** subscribes to both evaluation results, merges metrics/plots, attempts net-load if timestamps overlap, and writes a unified report.

### Important feasibility note about Net-Load with GEFCom datasets

Your current datasets are **GEFCom2014 Load** and **GEFCom2014 Solar** (separate competition tracks with different time periods and non-shared systems). In many real setups, net-load \(\hat{N}=\hat{L}-\hat{S}\) is defined when load and solar belong to the **same grid and aligned timestamps**. Your integration agent should therefore:
- compute net-load **only if** there is a non-empty timestamp intersection,
- otherwise still generate a joint report and clearly state “net-load not computed due to no overlap”.

This keeps your Phase C engineering intact and sets up the thesis argument that joint net-load evaluation requires a *joint dataset* (e.g., OPSD / ENTSO-E later).

## Transformer core implementation in PyTorch

### Design choices and how they map to PatchTST and iTransformer

- **PatchTST**: key idea is to segment the lookback sequence into *patches* that become transformer tokens; this reduces attention computation and preserves local semantics. citeturn2search2turn2search4  
- **iTransformer**: key idea is to invert tokenization so that **variates/features become tokens** and attention captures inter-variable correlations; this is useful for multivariate forecasting. citeturn1search0turn1search1  

The code below gives you both:
- `PatchTransformerForecaster` (PatchTST-like patch tokens)
- `ITransformerForecaster` (variate-token inversion)
- A training helper with early stopping and metric reporting

### Code: `src/models/transformers.py`

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

@dataclass(frozen=True)
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 0.0


class EarlyStopping:
    """
    Simple early stopping on validation loss.
    Saves best_state_dict in-memory; caller can persist.
    """
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best = float("inf")
        self.bad_epochs = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        improved = (self.best - val_loss) > self.cfg.min_delta
        if improved:
            self.best = val_loss
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False  # do not stop
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.cfg.patience


def _rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(y_pred, y_true))


# ----------------------------
# PatchTST-like Patch Forecaster
# ----------------------------

@dataclass(frozen=True)
class PatchTransformerConfig:
    input_dim: int                 # num_input_channels (features)
    context_length: int            # sequence_length
    pred_length: int               # horizon H
    target_dim: int = 1            # number of target channels predicted

    patch_length: int = 16
    patch_stride: int = 16

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    use_cls_token: bool = True

    # Optional time embedding
    use_hour_embedding: bool = False  # needs hour_idx input
    hour_emb_dim: int = 256           # should match d_model if added to tokens


class PatchEmbedding(nn.Module):
    """
    Convert [B, L, F] -> patch tokens [B, P, d_model]
    by chunking time dimension into patches.
    """
    def __init__(self, input_dim: int, patch_length: int, patch_stride: int, d_model: int):
        super().__init__()
        self.input_dim = input_dim
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.proj = nn.Linear(patch_length * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        B, L, Fdim = x.shape
        if Fdim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {Fdim}")

        if L < self.patch_length:
            raise ValueError(f"Sequence length {L} < patch_length {self.patch_length}")

        patches = x.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)
        # patches: [B, P, patch_length, F]
        B2, P, Plen, F2 = patches.shape
        patches = patches.contiguous().view(B2, P, Plen * F2)  # [B, P, patch_length*F]
        tokens = self.proj(patches)  # [B, P, d_model]
        return tokens


class PatchTransformerForecaster(nn.Module):
    """
    PatchTST-like transformer forecaster:
      - patches as tokens (local segments)
      - transformer encoder over patch tokens
      - head outputs [B, H, target_dim]
    """
    def __init__(self, cfg: PatchTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbedding(
            input_dim=cfg.input_dim,
            patch_length=cfg.patch_length,
            patch_stride=cfg.patch_stride,
            d_model=cfg.d_model,
        )

        # number of patches for allocated positional embedding
        num_patches = 1 + (cfg.context_length - cfg.patch_length) // cfg.patch_stride
        self.num_patches = num_patches

        self.use_cls = cfg.use_cls_token
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            pos_len = num_patches + 1
        else:
            self.cls_token = None
            pos_len = num_patches

        self.pos_emb = nn.Parameter(torch.zeros(1, pos_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.use_hour_emb = cfg.use_hour_embedding
        if self.use_hour_emb:
            # hour embeddings are added to patch tokens by averaging over patch hours
            self.hour_emb = nn.Embedding(24, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.pred_length * cfg.target_dim),
        )

    def forward(self, x: torch.Tensor, hour_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, L, F] where L=context_length, F=input_dim
        hour_idx (optional): [B, L] integers 0..23
        returns: [B, H, target_dim]
        """
        B, L, Fdim = x.shape
        if L != self.cfg.context_length:
            raise ValueError(f"Expected context_length={self.cfg.context_length}, got {L}")

        tokens = self.patch_embed(x)  # [B, P, d_model]

        if self.use_hour_emb:
            if hour_idx is None:
                raise ValueError("use_hour_embedding=True but hour_idx is None")
            # compute patch-level hour embeddings
            # hour_idx: [B, L] -> unfold -> [B, P, patch_length]
            h_patches = hour_idx.unfold(dimension=1, size=self.cfg.patch_length, step=self.cfg.patch_stride)
            # embed each hour and average within patch -> [B, P, d_model]
            h_emb = self.hour_emb(h_patches)        # [B, P, patch_length, d_model]
            h_emb = h_emb.mean(dim=2)               # [B, P, d_model]
            tokens = tokens + h_emb

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_emb[:, : tokens.shape[1], :]

        z = self.encoder(tokens)
        z = self.norm(z)

        if self.use_cls:
            summary = z[:, 0, :]               # [B, d_model]
        else:
            summary = z.mean(dim=1)            # [B, d_model]

        out = self.head(summary)               # [B, H*target_dim]
        out = out.view(B, self.cfg.pred_length, self.cfg.target_dim)
        return out


# ----------------------------
# iTransformer-like Forecaster
# ----------------------------

@dataclass(frozen=True)
class ITransformerConfig:
    input_dim: int
    context_length: int
    pred_length: int
    target_dim: int = 1

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1


class ITransformerForecaster(nn.Module):
    """
    iTransformer-inspired forecaster:
      - variates/features are tokens
      - each token receives an embedding of its time-series over context_length
      - transformer encoder attends across variates
      - pooled representation -> horizon outputs
    """
    def __init__(self, cfg: ITransformerConfig):
        super().__init__()
        self.cfg = cfg

        # project each feature's time series (length L) into d_model
        self.time_proj = nn.Linear(cfg.context_length, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.pred_length * cfg.target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, F]
        return: [B, H, target_dim]
        """
        B, L, Fdim = x.shape
        if L != self.cfg.context_length:
            raise ValueError(f"Expected context_length={self.cfg.context_length}, got {L}")
        if Fdim != self.cfg.input_dim:
            raise ValueError(f"Expected input_dim={self.cfg.input_dim}, got {Fdim}")

        # invert -> [B, F, L]
        x_inv = x.transpose(1, 2).contiguous()
        # embed each variate token -> [B, F, d_model]
        tokens = self.time_proj(x_inv)  # linear over last dimension L

        z = self.encoder(tokens)        # attention across feature tokens
        z = self.norm(z)
        summary = z.mean(dim=1)         # pool across features

        out = self.head(summary).view(B, self.cfg.pred_length, self.cfg.target_dim)
        return out


# ----------------------------
# Training loop helpers
# ----------------------------

@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    pin_memory: bool = True

    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig(patience=10, min_delta=0.0)


@torch.no_grad()
def predict_batches(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    use_hour_embedding: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds = []
    trues = []
    for batch in loader:
        if use_hour_embedding:
            x, y, hour_idx = batch
            x = x.to(device)
            y = y.to(device)
            hour_idx = hour_idx.to(device)
            yhat = model(x, hour_idx=hour_idx)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)

        preds.append(yhat.detach().cpu())
        trues.append(y.detach().cpu())

    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainerConfig,
    *,
    use_hour_embedding: bool = False,
) -> Dict[str, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    stopper = EarlyStopping(cfg.early_stopping)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            opt.zero_grad(set_to_none=True)

            if use_hour_embedding:
                x, y, hour_idx = batch
                x = x.to(device)
                y = y.to(device)
                hour_idx = hour_idx.to(device)
                yhat = model(x, hour_idx=hour_idx)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                yhat = model(x)

            loss = F.mse_loss(yhat, y)
            loss.backward()

            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1

        # validation
        with torch.no_grad():
            y_pred, y_true = predict_batches(model, val_loader, device, use_hour_embedding=use_hour_embedding)
            val_loss = float(F.mse_loss(y_pred, y_true).cpu())

        should_stop = stopper.step(val_loss, model)
        if should_stop:
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    return {"best_val_mse": float(stopper.best)}
```

## Updating the execution layer to stop flattening windows

Your Phase B `ModelDeveloperAgent` flattened `(X_hist, X_future)` into a single vector for sklearn/XGBoost. In Phase C:

- You must build a tensor `X_in` shaped **`[B, L, F]`**, where `L` is either:
  - Strategy A: `L = W` (history only), or  
  - Strategy B: `L = W + H` (history + future covariates “visible” to model)

PatchTST’s expected shape `(bs, sequence_length, num_input_channels)` still holds either way. citeturn0search1turn0search4  

### Code: `src/agents/model_developer_agent.py` (transformer-capable executor)

This is a **complete replacement** for your Phase B model developer (it can still keep the baseline path if you want, but here we focus on transformers).

```python
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loaders.windowing import SlidingWindowDataset, WindowConfig
from src.models.transformers import (
    PatchTransformerConfig, PatchTransformerForecaster,
    ITransformerConfig, ITransformerForecaster,
    TrainerConfig, train_model, predict_batches
)

from .base_agent import BaseAgent


@dataclass
class TransformerExecutionConfig:
    model_name: str             # "patchtst" | "itransformer"
    strategy: str               # "A" | "B"
    add_future_indicator: bool  # adds an extra channel marking future part


class TransformerInputDataset(Dataset):
    """
    Wrap SlidingWindowDataset to produce transformer-ready (X_in, Y) pairs.

    Base ds returns:
      X_hist:   [W, D_hist]
      X_future: [H, D_future]
      Y:        [H, D_y]

    We output:
      X_in: [L, F] where L=W (Strategy A) or L=W+H (Strategy B)
      Y:    [H, D_y]
    Optionally also output hour_idx: [L] for hour embeddings.
    """
    def __init__(
        self,
        base_ds: SlidingWindowDataset,
        hist_cols: List[str],
        future_cols: List[str],
        target_cols: List[str],
        strategy: str,
        add_future_indicator: bool,
        include_hour_idx: bool,
    ):
        self.ds = base_ds
        self.hist_cols = hist_cols
        self.future_cols = future_cols
        self.target_cols = target_cols
        self.strategy = strategy
        self.add_future_indicator = add_future_indicator
        self.include_hour_idx = include_hour_idx

        # Map future cols into hist col positions (so we can fill a full vector)
        self.hist_index = {c: i for i, c in enumerate(hist_cols)}
        self.future_to_hist = {j: self.hist_index[c] for j, c in enumerate(future_cols) if c in self.hist_index}

        self.W = base_ds.cfg.W
        self.H = base_ds.cfg.H

    def __len__(self) -> int:
        return len(self.ds)

    def _make_hour_idx(self, idx: int, L: int) -> torch.Tensor:
        # reconstruct future_start timestamp and build hour indices assuming hourly spacing
        zone, hs, he, fs, fe = self.ds.window_time_bounds(idx)
        # build time array length L: history then future
        # history ends at he, so history start = he - (W-1) hours
        hist_start = he - pd.Timedelta(hours=self.W - 1)
        times = pd.date_range(hist_start, periods=L, freq="h")
        hours = torch.tensor(times.hour.to_numpy(dtype=np.int64), dtype=torch.long)
        return hours

    def __getitem__(self, idx: int):
        x_hist, x_fut, y = self.ds[idx]  # tensors
        # x_hist: [W, Dh], x_fut:[H, Df], y:[H, Dy]
        W, Dh = x_hist.shape
        H, Df = x_fut.shape

        if self.strategy.upper() == "A":
            x_in = x_hist
        else:
            # Strategy B: build full sequence length W+H with feature dim Dh (hist_cols)
            # Fill history with x_hist
            x_full = torch.zeros((W + H, Dh), dtype=x_hist.dtype)
            x_full[:W, :] = x_hist

            # Fill future covariates into their positions
            for j in range(Df):
                if j in self.future_to_hist:
                    x_full[W:, self.future_to_hist[j]] = x_fut[:, j]

            # Optional future indicator channel
            if self.add_future_indicator:
                ind = torch.zeros((W + H, 1), dtype=x_hist.dtype)
                ind[W:, 0] = 1.0
                x_full = torch.cat([x_full, ind], dim=1)

            x_in = x_full

        if self.include_hour_idx:
            hour_idx = self._make_hour_idx(idx, x_in.shape[0])
            return x_in, y, hour_idx

        return x_in, y


class ModelDeveloperAgent(BaseAgent):
    """
    Executes transformer model training on GPU (if available),
    saves model + prediction artifacts, publishes evaluation results.
    """
    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        self.subscribe(topics=["train_execute"], start_from="latest")

        self.run_dir = Path("logs") / "runs" / self.run_id
        (self.run_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _inverse_targets(self, scaler_obj: Any, y_scaled: np.ndarray) -> np.ndarray:
        """
        scaler_obj is your LeakageSafeScaler (joblib-loaded). It inverse transforms
        targets in 2D form. We'll reshape [N, H, Dy] -> [N*H, Dy] -> inverse -> reshape back.
        """
        if not hasattr(scaler_obj, "inverse_transform_targets"):
            return y_scaled

        N, H, Dy = y_scaled.shape
        y2 = y_scaled.reshape(N * H, Dy)
        y2_inv = scaler_obj.inverse_transform_targets(y2)
        return y2_inv.reshape(N, H, Dy)

    def _metrics_and_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plot_path: Path,
        title: str,
        max_points: int = 400,
    ) -> Dict[str, float]:
        # flatten over horizons and outputs
        mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))
        rmse = float(math.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))))

        # plot horizon step 1 for readability
        y_t0 = y_true[:, 0, 0]
        y_p0 = y_pred[:, 0, 0]
        n = min(len(y_t0), max_points)

        plt.figure()
        plt.plot(y_t0[:n], label="Actual")
        plt.plot(y_p0[:n], label="Predicted")
        plt.title(title)
        plt.xlabel("Window index")
        plt.ylabel("Target (original units)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path.as_posix(), dpi=150)
        plt.close()

        return {"mae": mae, "rmse": rmse}

    def _save_predictions_parquet(
        self,
        base_ds: SlidingWindowDataset,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        out_path: Path,
        track: str,
        model_name: str,
    ) -> str:
        """
        Writes a long-format parquet:
          ZONEID, timestamp, horizon, y_true, y_pred, track, model
        using window_time_bounds() to reconstruct horizon timestamps (hourly spacing).
        """
        rows = []
        N, H, Dy = y_true.shape
        assert Dy == 1, "This helper assumes single-target; extend to multi-target if needed."

        for i in range(N):
            zone, hs, he, fs, fe = base_ds.window_time_bounds(i)
            times = pd.date_range(fs, periods=H, freq="h")
            for h in range(H):
                rows.append({
                    "ZONEID": int(zone),
                    "timestamp": times[h],
                    "horizon": int(h + 1),
                    "y_true": float(y_true[i, h, 0]),
                    "y_pred": float(y_pred[i, h, 0]),
                    "track": track,
                    "model": model_name,
                })

        dfp = pd.DataFrame(rows)
        dfp.to_parquet(out_path)
        return out_path.as_posix()

    def _build_model(self, model_name: str, input_dim: int, context_length: int, pred_length: int, use_hour_embedding: bool) -> torch.nn.Module:
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

        raise ValueError(f"Unknown transformer model: {model_name}")

    def tick(self) -> None:
        for msg in self.poll(max_messages=10):
            if msg.kind != "command":
                continue
            if msg.payload.get("command") != "train_execute":
                continue

            corr_id = msg.correlation_id
            try:
                track = msg.payload["track"]                        # "load" or "solar"
                paths = msg.payload["paths"]
                cols = msg.payload["columns"]

                exec_cfg = msg.payload["exec_cfg"]                  # dict
                model_name = exec_cfg["model_name"]
                strategy = exec_cfg["strategy"]
                add_future_indicator = bool(exec_cfg.get("add_future_indicator", True))
                use_hour_embedding = bool(exec_cfg.get("use_hour_embedding", False))

                window = msg.payload["window"]
                W = int(window["W"])
                H = int(window["H"])
                stride = int(window.get("stride", H))               # default stride=H

                train_params = msg.payload.get("training", {})
                batch_size = int(train_params.get("batch_size", 128))
                epochs = int(train_params.get("epochs", 50))
                lr = float(train_params.get("lr", 1e-3))
                patience = int(train_params.get("patience", 10))

                # Load data (scaled)
                df_train = pd.read_parquet(paths["train_scaled"])
                df_val = pd.read_parquet(paths["val_scaled"])
                df_test = pd.read_parquet(paths["test_scaled"])

                scaler_obj = joblib.load(paths["scaler"])

                base_cfg = WindowConfig(W=W, H=H, stride=stride)

                # Base window datasets
                ds_train_base = SlidingWindowDataset(df_train, cols["hist_cols"], cols["future_cols"], cols["target_cols"], base_cfg, return_times=False)
                ds_val_base   = SlidingWindowDataset(df_val,   cols["hist_cols"], cols["future_cols"], cols["target_cols"], base_cfg, return_times=False)
                ds_test_base  = SlidingWindowDataset(df_test,  cols["hist_cols"], cols["future_cols"], cols["target_cols"], base_cfg, return_times=False)

                # Derived transformer datasets
                ds_train = TransformerInputDataset(ds_train_base, cols["hist_cols"], cols["future_cols"], cols["target_cols"], strategy, add_future_indicator, include_hour_idx=use_hour_embedding)
                ds_val   = TransformerInputDataset(ds_val_base,   cols["hist_cols"], cols["future_cols"], cols["target_cols"], strategy, add_future_indicator, include_hour_idx=use_hour_embedding)
                ds_test  = TransformerInputDataset(ds_test_base,  cols["hist_cols"], cols["future_cols"], cols["target_cols"], strategy, add_future_indicator, include_hour_idx=use_hour_embedding)

                # Determine model input size and context length from dataset sample
                sample = ds_train[0]
                if use_hour_embedding:
                    x0, y0, h0 = sample
                else:
                    x0, y0 = sample
                context_length = x0.shape[0]
                input_dim = x0.shape[1]

                model = self._build_model(model_name, input_dim=input_dim, context_length=context_length, pred_length=H, use_hour_embedding=use_hour_embedding)

                # DataLoaders
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

                # Train
                tcfg = TrainerConfig(
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=1e-4,
                    grad_clip_norm=1.0,
                    early_stopping={"patience": patience, "min_delta": 0.0}  # handled below
                )  # NOTE: dataclass expects EarlyStoppingConfig; keep simple below.

                tcfg = TrainerConfig(
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=1e-4,
                    grad_clip_norm=1.0,
                    early_stopping=type(tcfg.early_stopping)(patience=patience, min_delta=0.0),
                )

                train_stats = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=self.device,
                    cfg=tcfg,
                    use_hour_embedding=use_hour_embedding and (model_name == "patchtst"),
                )

                # Predict on test
                y_pred_t, y_true_t = predict_batches(
                    model,
                    test_loader,
                    self.device,
                    use_hour_embedding=use_hour_embedding and (model_name == "patchtst"),
                )

                # Convert to numpy
                y_pred = y_pred_t.numpy()
                y_true = y_true_t.numpy()

                # Inverse scaling for reporting
                y_pred_inv = self._inverse_targets(scaler_obj, y_pred)
                y_true_inv = self._inverse_targets(scaler_obj, y_true)

                # Save model
                model_path = self.run_dir / "models" / f"{track}_{model_name}_{strategy}.pt"
                torch.save({"state_dict": model.state_dict()}, model_path)

                # Plot
                plot_path = self.run_dir / "plots" / f"{track}_{model_name}_{strategy}_pred_vs_actual.png"
                metrics = self._metrics_and_plot(
                    y_true=y_true_inv,
                    y_pred=y_pred_inv,
                    plot_path=plot_path,
                    title=f"{track.upper()} | {model_name} | Strategy {strategy} | Pred vs Actual (H=1)",
                )

                # Save predictions parquet
                pred_path = self.run_dir / "predictions" / f"{track}_{model_name}_{strategy}_predictions.parquet"
                pred_path_str = self._save_predictions_parquet(ds_test_base, y_true_inv, y_pred_inv, pred_path, track, model_name)

                # Publish evaluation result
                self.publish(
                    topic_type="evaluate",
                    kind="result",
                    payload={
                        "event": "evaluation_complete",
                        "track": track,
                        "model_name": model_name,
                        "strategy": strategy,
                        "metrics": metrics,
                        "train_stats": train_stats,
                        "artifacts": {
                            "model_path": model_path.as_posix(),
                            "plot_path": plot_path.as_posix(),
                            "predictions_path": pred_path_str,
                        },
                        "device": str(self.device),
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
                    },
                    correlation_id=corr_id,
                    level="error",
                )
```

### Why this is correct for transformers

- It feeds unflattened `(B, L, F)` tensors into transformer models, matching PatchTST’s shape expectation. citeturn0search1turn0search4  
- It uses PyTorch `DataLoader` batching behavior (batch dimension prepended) as documented. citeturn2search1turn2search5  
- It produces evaluation artifacts (model checkpoint, plot, and prediction parquet) that your Integration agent can consume.

## Specialized forecasting agents and Forecast Integration agent

### Code: `src/agents/load_forecasting_agent.py`

This agent “specializes” the config for load and hands off a `train_execute` command to the ModelDeveloper.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

from .base_agent import BaseAgent


@dataclass
class LoadTransformerDefaults:
    model_name: str = "patchtst"     # "patchtst" | "itransformer"
    strategy: str = "A"              # Load often works with history-only (A)
    add_future_indicator: bool = True
    use_hour_embedding: bool = False  # load already has sin/cos time features in many setups


class LoadForecastingAgent(BaseAgent):
    def __init__(self, cfg, pool, defaults: Optional[LoadTransformerDefaults] = None):
        super().__init__(cfg, pool)
        self.defaults = defaults or LoadTransformerDefaults()
        self.subscribe(topics=["train"], start_from="latest")

    def tick(self) -> None:
        for msg in self.poll(max_messages=20):
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
```

### Code: `src/agents/solar_forecasting_agent.py`

This agent enforces solar-specific logic: decumulated variable presence, NWP predictor selection, known-future covariates.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import pandas as pd

from .base_agent import BaseAgent


SOLAR_ACCUM_COLS = ["VAR169", "VAR175", "VAR178", "VAR228"]  # common accumulated fields in GEFCom solar pipelines


@dataclass
class SolarTransformerDefaults:
    model_name: str = "patchtst"
    strategy: str = "B"                 # Solar benefits from known-future NWP predictors
    add_future_indicator: bool = True
    use_hour_embedding: bool = True     # requirement: hour-of-day embeddings
    selected_predictors: Optional[List[str]] = None  # e.g., ["VAR78","VAR79",...]


class SolarForecastingAgent(BaseAgent):
    def __init__(self, cfg, pool, defaults: Optional[SolarTransformerDefaults] = None):
        super().__init__(cfg, pool)
        self.defaults = defaults or SolarTransformerDefaults()
        self.subscribe(topics=["train"], start_from="latest")

    def _select_predictors(self, columns: Dict[str, List[str]], selected: Optional[List[str]]) -> Dict[str, List[str]]:
        if not selected:
            return columns

        hist_cols = columns["hist_cols"]
        fut_cols = columns["future_cols"]

        # Keep POWER always, keep time features if present, and keep selected VARs
        keep = set(selected)
        new_hist = [c for c in hist_cols if (c == "POWER") or (c.startswith("hour_") or c.startswith("dow_") or c.startswith("month_")) or (c in keep)]
        new_fut = [c for c in fut_cols if (c.startswith("hour_") or c.startswith("dow_") or c.startswith("month_")) or (c in keep)]

        out = dict(columns)
        out["hist_cols"] = new_hist
        out["future_cols"] = new_fut
        return out

    def _assert_decumulated_vars_present(self, df_train_scaled_path: str, required: List[str]) -> None:
        # We cannot directly detect "decumulated vs not" by name (values differ),
        # but we can enforce these columns exist as a sanity check.
        df = pd.read_parquet(df_train_scaled_path, columns=[c for c in required if c is not None and c != ""])
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Solar required predictor '{c}' not present in processed data.")

    def tick(self) -> None:
        for msg in self.poll(max_messages=20):
            if msg.kind != "command":
                continue
            if msg.payload.get("command") != "train":
                continue
            if msg.payload.get("track") != "solar":
                continue

            corr_id = msg.correlation_id
            plan = msg.payload

            # Solar-specific: ensure accumulated vars exist (baseline sanity).
            # (Decumulation happens in Phase A; here we verify columns exist.)
            self._assert_decumulated_vars_present(plan["paths"]["train_scaled"], SOLAR_ACCUM_COLS)

            # Solar-specific: predictor selection (optional)
            cols = self._select_predictors(plan["columns"], plan.get("selected_predictors", self.defaults.selected_predictors))

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
```

### Code: `src/agents/forecast_integration_agent.py`

This agent merges load + solar evaluation artifacts into a single report and optionally computes net-load.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from .base_agent import BaseAgent


class ForecastIntegrationAgent(BaseAgent):
    """
    Subscribes to evaluate results from both tracks.
    Produces:
      - joint report (Markdown)
      - optional net-load predictions if timestamps overlap
    """
    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        self.subscribe(topics=["evaluate"], start_from="latest")

        self.run_dir = Path("logs") / "runs" / self.run_id
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "integrated").mkdir(parents=True, exist_ok=True)

        self._results: Dict[str, Dict[str, Any]] = {}

    def _try_compute_net_load(self) -> Optional[dict]:
        if "load" not in self._results or "solar" not in self._results:
            return None

        load_pred_path = self._results["load"]["artifacts"]["predictions_path"]
        solar_pred_path = self._results["solar"]["artifacts"]["predictions_path"]

        dfL = pd.read_parquet(load_pred_path)
        dfS = pd.read_parquet(solar_pred_path)

        # align on timestamp + horizon (and optionally ZONEID)
        key_cols = ["timestamp", "horizon"]
        if "ZONEID" in dfL.columns and "ZONEID" in dfS.columns:
            # solar has multiple zones; load often has one. We'll include ZONEID for strictness.
            key_cols = ["ZONEID"] + key_cols

        merged = dfL.merge(dfS, on=key_cols, suffixes=("_load", "_solar"))
        if len(merged) == 0:
            return {"computed": False, "reason": "No overlapping timestamps/horizons between load and solar predictions."}

        merged["net_true"] = merged["y_true_load"] - merged["y_true_solar"]
        merged["net_pred"] = merged["y_pred_load"] - merged["y_pred_solar"]

        out_path = self.run_dir / "integrated" / "net_load_predictions.parquet"
        merged.to_parquet(out_path)

        return {"computed": True, "path": out_path.as_posix(), "n_rows": int(len(merged))}

    def _write_report(self, net_info: Optional[dict]) -> str:
        report_path = self.run_dir / "reports" / "joint_decision_report.md"

        load = self._results.get("load")
        solar = self._results.get("solar")

        lines = []
        lines.append(f"# Joint Load + Solar Forecast Report ({self.run_id})\n")

        lines.append("## Load forecasting result\n")
        if load:
            lines.append(f"- Model: `{load.get('model_name')}` | Strategy: `{load.get('strategy')}`\n")
            lines.append(f"- MAE: **{load['metrics']['mae']:.6f}**, RMSE: **{load['metrics']['rmse']:.6f}**\n")
            lines.append(f"- Plot: `{load['artifacts']['plot_path']}`\n")
            lines.append(f"- Predictions: `{load['artifacts']['predictions_path']}`\n")
        else:
            lines.append("- Not available.\n")

        lines.append("\n## Solar forecasting result\n")
        if solar:
            lines.append(f"- Model: `{solar.get('model_name')}` | Strategy: `{solar.get('strategy')}`\n")
            lines.append(f"- MAE: **{solar['metrics']['mae']:.6f}**, RMSE: **{solar['metrics']['rmse']:.6f}**\n")
            lines.append(f"- Plot: `{solar['artifacts']['plot_path']}`\n")
            lines.append(f"- Predictions: `{solar['artifacts']['predictions_path']}`\n")
        else:
            lines.append("- Not available.\n")

        lines.append("\n## Net-load integration\n")
        if not net_info:
            lines.append("- Net-load not computed (missing one of the tracks).\n")
        else:
            if not net_info.get("computed", False):
                lines.append(f"- Net-load not computed: {net_info.get('reason')}\n")
            else:
                lines.append(f"- Net-load computed ✅\n")
                lines.append(f"- Rows: {net_info.get('n_rows')}\n")
                lines.append(f"- Artifact: `{net_info.get('path')}`\n")

        report_path.write_text("".join(lines), encoding="utf-8")
        return report_path.as_posix()

    def tick(self) -> None:
        for msg in self.poll(max_messages=50):
            if msg.kind != "result":
                continue
            if msg.payload.get("event") != "evaluation_complete":
                continue

            track = msg.payload.get("track")
            if track not in ("load", "solar"):
                continue

            # Cache
            self._results[track] = {
                "track": track,
                "model_name": msg.payload.get("model_name"),
                "strategy": msg.payload.get("strategy"),
                "metrics": msg.payload.get("metrics", {}),
                "artifacts": msg.payload.get("artifacts", {}),
            }

        # If both arrived, integrate once
        if "load" in self._results and "solar" in self._results and not self._results.get("_integrated"):
            net_info = self._try_compute_net_load()
            report_path = self._write_report(net_info)
            self._results["_integrated"] = True

            self.publish(
                topic_type="integrate",
                kind="result",
                payload={
                    "event": "joint_report_ready",
                    "report_path": report_path,
                    "net_load": net_info or {"computed": False, "reason": "missing tracks"},
                },
            )
```

## Updated orchestrator for parallel tracks

### Code: `orchestrator.py` (root) with `--mode research`

This orchestrator runs both tracks in parallel and stops when the Integration agent publishes `joint_report_ready`.

```python
from __future__ import annotations

import argparse
import time
from datetime import datetime

from src.agents.message_pool import MessagePool
from src.agents.base_agent import AgentConfig
from src.agents.task_manager_agent import TaskManagerAgent, RunConfig
from src.agents.preparation_agent import PreparationAgent
from src.agents.model_manager_agent import ModelManagerAgent
from src.agents.model_developer_agent import ModelDeveloperAgent

from src.agents.load_forecasting_agent import LoadForecastingAgent
from src.agents.solar_forecasting_agent import SolarForecastingAgent
from src.agents.forecast_integration_agent import ForecastIntegrationAgent


def make_run_id() -> str:
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


def parse_args():
    p = argparse.ArgumentParser(description="Phase C Orchestrator (research mode): load + solar transformers + joint report")
    p.add_argument("--mode", type=str, choices=["baseline", "research"], default="research")
    p.add_argument("--manifest", type=str, default="data/processed/dataset_manifest.json")

    # Windowing
    p.add_argument("--W", type=int, default=168)
    p.add_argument("--H", type=int, default=24)
    p.add_argument("--stride", type=int, default=24)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)

    # Models
    p.add_argument("--load_model", type=str, choices=["patchtst", "itransformer"], default="patchtst")
    p.add_argument("--solar_model", type=str, choices=["patchtst", "itransformer"], default="patchtst")

    p.add_argument("--tick_seconds", type=float, default=1.0)
    p.add_argument("--timeout_seconds", type=float, default=7200.0)
    return p.parse_args()


def main():
    args = parse_args()
    run_id = make_run_id()

    pool = MessagePool(log_path="logs/message_pool.jsonl", load_existing=True)

    # TaskManager publishes prepares for both tracks (you can implement this by running TaskManager twice
    # or by enhancing TaskManager to accept multi-track configs; simplest is two TaskManagers).
    # We'll do two TaskManagers (load + solar).

    common_window = {"W": args.W, "H": args.H, "stride": args.stride}
    common_training = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "patience": args.patience}

    # Use two task managers to emit two prepare commands in the same run.
    tm_load = TaskManagerAgent(
        AgentConfig(agent_id="TaskManagerLoad", role="task_manager", run_id=run_id),
        pool,
        RunConfig(
            manifest_path=args.manifest,
            track="load",
            model_type="transformer",
            window=common_window,
            training=common_training,
            hpo={"enabled": False, "n_trials": 1},
        ),
    )

    tm_solar = TaskManagerAgent(
        AgentConfig(agent_id="TaskManagerSolar", role="task_manager", run_id=run_id),
        pool,
        RunConfig(
            manifest_path=args.manifest,
            track="solar",
            model_type="transformer",
            window=common_window,
            training=common_training,
            hpo={"enabled": False, "n_trials": 1},
        ),
    )

    prep = PreparationAgent(AgentConfig(agent_id="PreparationAgent", role="preparation", run_id=run_id), pool)

    # NOTE: you should extend ModelManagerAgent so that on READY it publishes train commands for transformer mode.
    # For now, we reuse ModelManagerAgent and rely on forecasting agents to interpret the train payload.
    mm = ModelManagerAgent(AgentConfig(agent_id="ModelManagerAgent", role="model_manager", run_id=run_id), pool)

    # New specialized forecasting agents
    load_agent = LoadForecastingAgent(AgentConfig(agent_id="LoadForecastingAgent", role="load_forecasting", run_id=run_id), pool)
    solar_agent = SolarForecastingAgent(AgentConfig(agent_id="SolarForecastingAgent", role="solar_forecasting", run_id=run_id), pool)

    # Executor
    dev = ModelDeveloperAgent(AgentConfig(agent_id="ModelDeveloperAgent", role="model_developer", run_id=run_id), pool)

    # Integration
    integ = ForecastIntegrationAgent(AgentConfig(agent_id="ForecastIntegrationAgent", role="forecast_integration", run_id=run_id), pool)

    agents = [tm_load, tm_solar, prep, mm, load_agent, solar_agent, dev, integ]

    # In research mode, publish two transformer train commands once both tracks are ready.
    # Easiest: orchestrator monitors ready events and injects train plans directly.
    ready = {"load": False, "solar": False}
    start = time.time()
    done = False

    while not done and (time.time() - start) < args.timeout_seconds:
        for a in agents:
            a.tick()

        # Detect READY events
        prep_msgs = pool.find_by_topic("prepare", topic_source=run_id)
        for m in prep_msgs:
            if m.kind == "event" and m.payload.get("event") == "ready":
                ready[m.payload.get("track")] = True

        # When both ready: inject track-specific train commands once
        if args.mode == "research" and ready["load"] and ready["solar"]:
            # Only inject once
            already = any(m.topic.startswith(f"train/{run_id}") and m.kind == "command" for m in pool.find_by_topic("train", topic_source=run_id))
            if not already:
                # Load plan
                load_ready_msg = next(m for m in prep_msgs if m.kind == "event" and m.payload.get("event") == "ready" and m.payload.get("track") == "load")
                pool.publish(
                    __import__("src.agents.messages", fromlist=["Message", "TopicId"]).Message.new(
                        topic=__import__("src.agents.messages", fromlist=["TopicId"]).TopicId("train", run_id),
                        kind="command",
                        sender="Orchestrator",
                        sender_role="system",
                        payload={
                            "command": "train",
                            "track": "load",
                            "paths": load_ready_msg.payload["paths"],
                            "columns": load_ready_msg.payload["columns"],
                            "window": common_window,
                            "training": common_training,
                            "model_name": args.load_model,
                            "strategy": "A",
                            "add_future_indicator": True,
                            "use_hour_embedding": False,
                        },
                        run_id=run_id,
                    )
                )

                # Solar plan
                solar_ready_msg = next(m for m in prep_msgs if m.kind == "event" and m.payload.get("event") == "ready" and m.payload.get("track") == "solar")
                pool.publish(
                    __import__("src.agents.messages", fromlist=["Message", "TopicId"]).Message.new(
                        topic=__import__("src.agents.messages", fromlist=["TopicId"]).TopicId("train", run_id),
                        kind="command",
                        sender="Orchestrator",
                        sender_role="system",
                        payload={
                            "command": "train",
                            "track": "solar",
                            "paths": solar_ready_msg.payload["paths"],
                            "columns": solar_ready_msg.payload["columns"],
                            "window": common_window,
                            "training": common_training,
                            "model_name": args.solar_model,
                            "strategy": "B",
                            "add_future_indicator": True,
                            "use_hour_embedding": True,
                            # optional predictor selection:
                            # "selected_predictors": ["VAR78","VAR79","VAR134","VAR157","VAR164","VAR165","VAR166","VAR167","VAR169","VAR175","VAR178","VAR228"]
                        },
                        run_id=run_id,
                    )
                )

        # Stop when integration result arrives
        integ_msgs = pool.find_by_topic("integrate", topic_source=run_id)
        if any(m.kind == "result" and m.payload.get("event") == "joint_report_ready" for m in integ_msgs):
            done = True

        time.sleep(args.tick_seconds)

    if not done:
        print(f"[WARN] Timeout without joint report. run_id={run_id}")
    else:
        print(f"[OK] Research run completed. run_id={run_id}")
        print(f"Check logs/runs/{run_id}/reports/joint_decision_report.md")
        print("Message trace: logs/message_pool.jsonl")


if __name__ == "__main__":
    main()
```

## Optional probabilistic forecasting skeleton for thesis feasibility

If you decide to add probabilistic outputs (quantiles) later, the key is to track **pinball loss**, which is the standard quantile regression loss and is explicitly provided in scikit-learn as `mean_pinball_loss`. citeturn3search0turn3search5  

A practical Phase C “skeleton” approach:
- change your transformer head to output multiple quantiles per horizon (e.g., 0.1, 0.5, 0.9),
- compute pinball loss during validation and store it in the evaluation message.

You can use scikit-learn’s documented pinball loss definition and metric to keep evaluation standardized. citeturn3search0turn3search5  

## What “success” looks like when you run Phase C

When you run:

```bash
python orchestrator.py --mode research
```

You should get:

- `logs/message_pool.jsonl` containing a full trail:
  - prepare commands for load and solar
  - ready events
  - train commands
  - train_execute commands
  - evaluate results for both tracks
  - integrate result with a report path
- Artifacts under:
  - `logs/runs/<run_id>/models/`
  - `logs/runs/<run_id>/plots/`
  - `logs/runs/<run_id>/predictions/`
  - `logs/runs/<run_id>/reports/joint_decision_report.md`

This completes the Phase C Step One deliverables: transformer core integration, specialized forecasting agents, and a working joint report generation pipeline.
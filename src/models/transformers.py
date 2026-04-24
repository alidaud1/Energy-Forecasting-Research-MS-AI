from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 0.0


class EarlyStopping:
    """
    Early stopping on validation loss.
    Stores best_state_dict in-memory; caller may persist later.
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
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.cfg.patience


def _rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(y_pred, y_true))


@dataclass(frozen=True)
class PatchTransformerConfig:
    input_dim: int
    context_length: int
    pred_length: int
    target_dim: int = 1

    patch_length: int = 16
    patch_stride: int = 16

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    use_cls_token: bool = True

    use_hour_embedding: bool = False
    hour_emb_dim: int = 256


class PatchEmbedding(nn.Module):
    """
    Convert [B, L, F] -> [B, P, d_model] by patching time dimension.
    """

    def __init__(self, input_dim: int, patch_length: int, patch_stride: int, d_model: int):
        super().__init__()
        self.input_dim = input_dim
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.proj = nn.Linear(patch_length * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Fdim = x.shape
        if Fdim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {Fdim}")
        if L < self.patch_length:
            raise ValueError(f"Sequence length {L} < patch_length {self.patch_length}")

        patches = x.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)
        # [B, P, patch_length, F]
        B2, P, Plen, F2 = patches.shape
        patches = patches.contiguous().view(B2, P, Plen * F2)
        return self.proj(patches)


class PatchTransformerForecaster(nn.Module):
    """
    PatchTST-like encoder-only forecaster:
    - patch tokens + optional CLS
    - transformer encoder
    - pooled summary -> [B, H, target_dim]
    """

    def __init__(self, cfg: PatchTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbedding(cfg.input_dim, cfg.patch_length, cfg.patch_stride, cfg.d_model)

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
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.pred_length * cfg.target_dim))

    def forward(self, x: torch.Tensor, hour_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, Fdim = x.shape
        if L != self.cfg.context_length:
            raise ValueError(f"Expected context_length={self.cfg.context_length}, got {L}")

        tokens = self.patch_embed(x)  # [B, P, d_model]

        if self.use_hour_emb:
            if hour_idx is None:
                raise ValueError("use_hour_embedding=True but hour_idx is None")
            h_patches = hour_idx.unfold(dimension=1, size=self.cfg.patch_length, step=self.cfg.patch_stride)
            h_emb = self.hour_emb(h_patches).mean(dim=2)
            tokens = tokens + h_emb

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = tokens + self.pos_emb[:, : tokens.shape[1], :]
        z = self.encoder(tokens)
        z = self.norm(z)

        summary = z[:, 0, :] if self.use_cls else z.mean(dim=1)
        out = self.head(summary).view(B, self.cfg.pred_length, self.cfg.target_dim)
        return out


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
    iTransformer-inspired encoder-only forecaster:
    - variates/features are tokens
    - each token embeds its time series (length L) -> d_model
    - attention across variates
    """

    def __init__(self, cfg: ITransformerConfig):
        super().__init__()
        self.cfg = cfg

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
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.pred_length * cfg.target_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Fdim = x.shape
        if L != self.cfg.context_length:
            raise ValueError(f"Expected context_length={self.cfg.context_length}, got {L}")
        if Fdim != self.cfg.input_dim:
            raise ValueError(f"Expected input_dim={self.cfg.input_dim}, got {Fdim}")

        x_inv = x.transpose(1, 2).contiguous()  # [B, F, L]
        tokens = self.time_proj(x_inv)  # [B, F, d_model]
        z = self.encoder(tokens)
        z = self.norm(z)
        summary = z.mean(dim=1)
        out = self.head(summary).view(B, self.cfg.pred_length, self.cfg.target_dim)
        return out


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
    on_epoch_end: Optional[Callable[[int, float, float], Any]] = None,
) -> Dict[str, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    stopper = EarlyStopping(cfg.early_stopping)

    history: Dict[str, List[float]] = {"train_mse": [], "val_mse": []}

    for epoch in range(cfg.epochs):
        model.train()
        train_num = 0.0
        train_den = 0.0
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
            # Weight by element count so epoch-average reflects variable batch sizes.
            n_el = float(y.numel())
            train_num += float(loss.detach().cpu()) * n_el
            train_den += n_el
            loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            opt.step()

        # validation
        y_pred, y_true = predict_batches(model, val_loader, device, use_hour_embedding=use_hour_embedding)
        val_loss = float(F.mse_loss(y_pred, y_true).cpu())
        train_loss = float(train_num / max(1.0, train_den))
        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)

        if on_epoch_end is not None:
            try:
                on_epoch_end(epoch, train_loss, val_loss)
            except Exception:
                # Training must not crash due to telemetry callbacks.
                pass
        if stopper.step(val_loss, model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return {"best_val_mse": float(stopper.best), "history": history}  # type: ignore[return-value]

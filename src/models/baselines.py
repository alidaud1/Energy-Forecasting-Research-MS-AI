from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


class MLPBaseline(nn.Module):
    def __init__(self, *, input_dim: int, output_dim: int, hidden_layers: int, hidden_size: int, dropout: float):
        super().__init__()
        hs = int(hidden_size)
        layers = []
        d = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(d, hs))
            layers.append(nn.ReLU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = hs
        layers.append(nn.Linear(d, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNNBaseline(nn.Module):
    def __init__(
        self,
        *,
        cell: str,
        hist_dim: int,
        fut_dim: int,
        hidden_size: int,
        layers: int,
        fc_size: int,
        output_dim: int,
    ):
        super().__init__()
        cell_l = str(cell or "").lower()
        rnn_cls = nn.LSTM if cell_l == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=int(hist_dim),
            hidden_size=int(hidden_size),
            num_layers=int(layers),
            batch_first=True,
        )
        self.fc1 = nn.Linear(int(hidden_size) + int(fut_dim), int(fc_size))
        self.fc2 = nn.Linear(int(fc_size), int(output_dim))
        self.act = nn.ReLU()

    def forward(self, x_hist: torch.Tensor, x_fut_flat: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x_hist)
        h = out[:, -1, :]
        z = torch.cat([h, x_fut_flat], dim=1)
        return self.fc2(self.act(self.fc1(z)))


class CNNBaseline(nn.Module):
    def __init__(
        self,
        *,
        hist_dim: int,
        fut_dim: int,
        filters: int,
        kernel_size: int,
        layers: int,
        fc_size: int,
        output_dim: int,
    ):
        super().__init__()
        c = int(filters)
        k = int(kernel_size)
        n_layers = int(layers)
        convs = []
        in_ch = int(hist_dim)
        for _ in range(max(1, n_layers)):
            convs.append(nn.Conv1d(in_ch, c, kernel_size=k, padding="same"))
            convs.append(nn.ReLU())
            in_ch = c
        self.conv = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average over time
        self.fc1 = nn.Linear(c + int(fut_dim), int(fc_size))
        self.fc2 = nn.Linear(int(fc_size), int(output_dim))
        self.act = nn.ReLU()

    def forward(self, x_hist: torch.Tensor, x_fut_flat: torch.Tensor) -> torch.Tensor:
        # x_hist: [N, W, D] -> conv expects [N, D, W]
        x = x_hist.transpose(1, 2)
        y = self.conv(x)
        y = self.pool(y).squeeze(-1)  # [N, C]
        z = torch.cat([y, x_fut_flat], dim=1)
        return self.fc2(self.act(self.fc1(z)))


@dataclass
class TorchTrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 3


def train_torch_regressor(
    *,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    cfg: TorchTrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, float, float]:
    """
    Minimal, production-safe training loop with early stopping on validation MSE.
    Returns: (best_model, best_val_mse, last_train_mse)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    if x_val is not None and y_val is not None:
        x_val = x_val.to(device)
        y_val = y_val.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    loss_fn = nn.MSELoss()

    n = x_train.shape[0]
    bs = int(max(16, cfg.batch_size))
    best_state = None
    best_val = float("inf")
    patience_left = int(max(1, cfg.patience))
    last_train = float("inf")

    for _epoch in range(int(max(1, cfg.epochs))):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        seen = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = x_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(xb.shape[0])
            seen += int(xb.shape[0])
        last_train = total / max(1, seen)

        if x_val is None or y_val is None:
            # No validation; keep latest.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = last_train
            continue

        model.eval()
        with torch.no_grad():
            pred = model(x_val)
            val = float(loss_fn(pred, y_val).detach().cpu())

        if val < best_val - 1e-8:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(max(1, cfg.patience))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val), float(last_train)


def train_torch_seq_regressor(
    *,
    model: nn.Module,
    x_hist_train: torch.Tensor,
    x_fut_train: torch.Tensor,
    y_train: torch.Tensor,
    x_hist_val: Optional[torch.Tensor] = None,
    x_fut_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    cfg: TorchTrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, float, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    x_hist_train = x_hist_train.to(device)
    x_fut_train = x_fut_train.to(device)
    y_train = y_train.to(device)
    if x_hist_val is not None and x_fut_val is not None and y_val is not None:
        x_hist_val = x_hist_val.to(device)
        x_fut_val = x_fut_val.to(device)
        y_val = y_val.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    loss_fn = nn.MSELoss()

    n = x_hist_train.shape[0]
    bs = int(max(16, cfg.batch_size))
    best_state = None
    best_val = float("inf")
    patience_left = int(max(1, cfg.patience))
    last_train = float("inf")

    for _epoch in range(int(max(1, cfg.epochs))):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        seen = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xh = x_hist_train.index_select(0, idx)
            xf = x_fut_train.index_select(0, idx)
            yb = y_train.index_select(0, idx)
            pred = model(xh, xf)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(xh.shape[0])
            seen += int(xh.shape[0])
        last_train = total / max(1, seen)

        if x_hist_val is None or x_fut_val is None or y_val is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = last_train
            continue

        model.eval()
        with torch.no_grad():
            pred = model(x_hist_val, x_fut_val)
            val = float(loss_fn(pred, y_val).detach().cpu())

        if val < best_val - 1e-8:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(max(1, cfg.patience))
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val), float(last_train)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x)).float()


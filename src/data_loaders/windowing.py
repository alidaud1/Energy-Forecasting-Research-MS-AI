from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowConfig:
    W: int  # history length
    H: int  # horizon length
    stride: int = 1


class SlidingWindowDataset(Dataset):
    """
    Torch Dataset that returns:
      X_hist:   [W, D_hist]
      X_future: [H, D_fut]
      Y:        [H, D_y]
    Optionally returns timestamp bounds for debugging/testing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        hist_cols: List[str],
        future_cols: List[str],
        target_cols: List[str],
        cfg: WindowConfig,
        return_times: bool = False,
    ):
        if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ZONEID", "timestamp"]:
            raise ValueError("Expected df indexed by MultiIndex ['ZONEID', 'timestamp']")
        if cfg.W <= 0 or cfg.H <= 0 or cfg.stride <= 0:
            raise ValueError("W, H, stride must be positive")

        self.df = df.sort_index()
        self.hist_cols = hist_cols
        self.future_cols = future_cols
        self.target_cols = target_cols
        self.cfg = cfg
        self.return_times = return_times

        self._zone_cache = {}
        self._window_map: List[Tuple[int, int]] = []

        for zone, g in self.df.groupby(level=0, sort=False):
            g = g.droplevel(0).sort_index()
            ts = g.index.to_numpy()
            x_hist_arr = g[self.hist_cols].to_numpy(dtype=np.float32)
            x_fut_arr = g[self.future_cols].to_numpy(dtype=np.float32)
            y_arr = g[self.target_cols].to_numpy(dtype=np.float32)

            n = len(g)
            max_start = n - (cfg.W + cfg.H)
            if max_start < 0:
                continue

            self._zone_cache[zone] = {"ts": ts, "x_hist": x_hist_arr, "x_fut": x_fut_arr, "y": y_arr}
            for start in range(0, max_start + 1, cfg.stride):
                self._window_map.append((zone, start))

        if len(self._window_map) == 0:
            raise ValueError("No windows available. Check W/H vs data length.")

    def __len__(self) -> int:
        return len(self._window_map)

    def window_time_bounds(self, idx: int):
        zone, start = self._window_map[idx]
        cache = self._zone_cache[zone]
        ts = cache["ts"]
        W, H = self.cfg.W, self.cfg.H
        hist_start = pd.Timestamp(ts[start])
        hist_end = pd.Timestamp(ts[start + W - 1])
        fut_start = pd.Timestamp(ts[start + W])
        fut_end = pd.Timestamp(ts[start + W + H - 1])
        return zone, hist_start, hist_end, fut_start, fut_end

    def __getitem__(self, idx: int):
        zone, start = self._window_map[idx]
        cache = self._zone_cache[zone]
        W, H = self.cfg.W, self.cfg.H

        X_hist = cache["x_hist"][start : start + W]
        X_future = cache["x_fut"][start + W : start + W + H]
        Y = cache["y"][start + W : start + W + H]

        X_hist_t = torch.from_numpy(X_hist)
        X_future_t = torch.from_numpy(X_future)
        Y_t = torch.from_numpy(Y)

        if self.return_times:
            _, hs, he, fs, fe = self.window_time_bounds(idx)
            return X_hist_t, X_future_t, Y_t, (hs, he, fs, fe)

        return X_hist_t, X_future_t, Y_t


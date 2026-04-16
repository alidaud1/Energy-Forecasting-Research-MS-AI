from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ScalerKind = Literal["standard", "minmax"]


@dataclass
class LeakageSafeScaler:
    feature_cols: List[str]
    target_cols: List[str]
    kind: ScalerKind = "standard"

    def __post_init__(self):
        if self.kind == "standard":
            self._x_scaler = StandardScaler()
            self._y_scaler = StandardScaler()
        elif self.kind == "minmax":
            self._x_scaler = MinMaxScaler()
            self._y_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler kind: {self.kind}")

        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "LeakageSafeScaler":
        x = train_df[self.feature_cols].to_numpy(dtype=np.float32)
        y = train_df[self.target_cols].to_numpy(dtype=np.float32)
        self._x_scaler.fit(x)
        self._y_scaler.fit(y)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit(train_df) first.")

        out = df.copy()
        out[self.feature_cols] = self._x_scaler.transform(out[self.feature_cols].to_numpy(dtype=np.float32))
        out[self.target_cols] = self._y_scaler.transform(out[self.target_cols].to_numpy(dtype=np.float32))
        return out

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit(train_df) first.")
        return self._y_scaler.inverse_transform(y_scaled)


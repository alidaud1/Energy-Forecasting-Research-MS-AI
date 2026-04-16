import numpy as np
import pandas as pd

from src.data_loaders.splits import chronological_split_by_zone
from src.data_loaders.windowing import SlidingWindowDataset, WindowConfig


def test_chronological_split_and_no_leakage():
    idx = pd.date_range("2023-01-01", periods=200, freq="h")
    df_zone1 = pd.DataFrame(
        {"feat": np.arange(len(idx), dtype=np.float32), "target": np.arange(len(idx), dtype=np.float32) * 2.0},
        index=idx,
    )
    df_zone1["ZONEID"] = 1
    df_zone1 = df_zone1.reset_index(names="timestamp").set_index(["ZONEID", "timestamp"])

    df_zone2 = df_zone1.copy()
    df_zone2 = df_zone2.reset_index()
    df_zone2["ZONEID"] = 2
    df_zone2 = df_zone2.set_index(["ZONEID", "timestamp"])

    df = pd.concat([df_zone1, df_zone2]).sort_index()

    split = chronological_split_by_zone(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

    for zone, g_tr in split.train.groupby(level=0):
        tr_max = g_tr.droplevel(0).index.max()
        va_min = split.val.xs(zone, level=0).index.min()
        te_min = split.test.xs(zone, level=0).index.min()
        assert tr_max < va_min
        assert va_min < te_min

    cfg = WindowConfig(W=24, H=12, stride=3)
    ds = SlidingWindowDataset(
        split.train,
        hist_cols=["feat", "target"],
        future_cols=["feat"],
        target_cols=["target"],
        cfg=cfg,
        return_times=True,
    )

    for i in [0, len(ds) // 3, len(ds) - 1]:
        _, _, _, (hs, he, fs, fe) = ds[i]
        assert he < fs, "History end must be strictly before future start (no leakage)"
        assert (fs - he) == pd.Timedelta(hours=1), "Future must start exactly 1 hour after hist end"
        assert fe >= fs


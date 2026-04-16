from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronoSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split_by_zone(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
) -> ChronoSplit:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != ["ZONEID", "timestamp"]:
        raise ValueError("Expected df indexed by MultiIndex ['ZONEID', 'timestamp']")

    trains, vals, tests = [], [], []
    for zone, g in df.groupby(level=0, sort=False):
        g = g.droplevel(0).sort_index()
        n = len(g)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_g = g.iloc[:n_train]
        val_g = g.iloc[n_train:n_train + n_val]
        test_g = g.iloc[n_train + n_val:]

        for part, bucket in [(train_g, trains), (val_g, vals), (test_g, tests)]:
            part = part.copy()
            part["ZONEID"] = zone
            part = part.reset_index(names="timestamp").set_index(["ZONEID", "timestamp"])
            bucket.append(part)

    return ChronoSplit(
        train=pd.concat(trains).sort_index(),
        val=pd.concat(vals).sort_index(),
        test=pd.concat(tests).sort_index(),
    )


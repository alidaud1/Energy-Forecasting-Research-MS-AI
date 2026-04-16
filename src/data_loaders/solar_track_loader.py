from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .io_utils import read_csv_from_root
from .time_parsing import parse_gefcom_solar_timestamps
from .solar_features import decumulate_solar_accumulators, SOLAR_ACCUMULATED_DEFAULT

PathLike = Union[str, Path]


@dataclass(frozen=True)
class SolarLoaderConfig:
    zip_path: Path
    task_id: int = 15
    apply_decumulation: bool = True
    accumulated_cols: Optional[list[str]] = None
    interpolate_missing_timestamps: bool = True
    freq: str = "h"


def load_gefcom_solar_task(config: SolarLoaderConfig) -> pd.DataFrame:
    """
    Returns a MultiIndex dataframe with index levels [ZONEID, timestamp].
    Columns include POWER (target) plus VAR features.
    """
    t = int(config.task_id)
    # Dataset uses no zero padding for train/predictors filenames (train1.csv, predictors1.csv, ...).
    train_path = f"Solar/Task {t}/train{t}.csv"
    pred_path = f"Solar/Task {t}/predictors{t}.csv"

    train_df = read_csv_from_root(config.zip_path, train_path)
    pred_df = read_csv_from_root(config.zip_path, pred_path)

    for d in (train_df, pred_df):
        if "ZONEID" not in d.columns or "TIMESTAMP" not in d.columns:
            raise ValueError(f"Unexpected solar schema: {d.columns.tolist()}")

    train_ts = parse_gefcom_solar_timestamps(train_df["TIMESTAMP"].astype(str).tolist())
    pred_ts = parse_gefcom_solar_timestamps(pred_df["TIMESTAMP"].astype(str).tolist())

    train_df = train_df.drop(columns=["TIMESTAMP"])
    pred_df = pred_df.drop(columns=["TIMESTAMP"])
    train_df.insert(1, "timestamp", train_ts)
    pred_df.insert(1, "timestamp", pred_ts)

    merged = pd.merge(
        pred_df,
        train_df[["ZONEID", "timestamp", "POWER"]],
        on=["ZONEID", "timestamp"],
        how="left",
        suffixes=("", "_y"),
        validate="many_to_one",
    )

    if "POWER" in pred_df.columns:
        if "POWER_y" in merged.columns:
            pred_power = merged["POWER"]
            train_power = merged["POWER_y"]
            mismatch = (
                pred_power.notnull()
                & train_power.notnull()
                & (np.abs(pred_power - train_power) > 1e-6)
            ).mean()
            if mismatch > 0:
                pass
            merged = merged.drop(columns=["POWER"])
            merged = merged.rename(columns={"POWER_y": "POWER"})
    elif "POWER_y" in merged.columns:
        merged = merged.rename(columns={"POWER_y": "POWER"})

    merged = merged.sort_values(["ZONEID", "timestamp"]).set_index(["ZONEID", "timestamp"]).sort_index()

    if config.apply_decumulation:
        cols = config.accumulated_cols or SOLAR_ACCUMULATED_DEFAULT
        merged = decumulate_solar_accumulators(merged, cols=cols)

    if config.interpolate_missing_timestamps:
        numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        interp_cols = [c for c in numeric_cols if c != "POWER"]

        out = []
        for zone, g in merged.groupby(level=0, sort=False):
            g = g.droplevel(0).sort_index()
            full_idx = pd.date_range(g.index.min(), g.index.max(), freq=config.freq)
            g2 = g.reindex(full_idx)
            if interp_cols:
                g2[interp_cols] = g2[interp_cols].interpolate(method="time", limit_direction="both")
            g2["ZONEID"] = zone
            g2 = g2.reset_index(names="timestamp").set_index(["ZONEID", "timestamp"])
            out.append(g2)

        merged = pd.concat(out).sort_index()

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load GEFCom2014 solar track and merge train + predictors.")
    parser.add_argument("--zip_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=15)
    parser.add_argument("--out_parquet", type=str, default="")
    args = parser.parse_args()

    cfg = SolarLoaderConfig(zip_path=Path(args.zip_path), task_id=args.task_id)
    df = load_gefcom_solar_task(cfg)
    print(df.head())
    print(df.tail())
    print("Shape:", df.shape)

    if args.out_parquet:
        df.to_parquet(args.out_parquet)
        print("Saved:", args.out_parquet)

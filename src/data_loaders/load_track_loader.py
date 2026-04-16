from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import warnings

from .io_utils import read_csv_from_root
from .time_parsing import parse_gefcom_load_timestamps

PathLike = Union[str, Path]


@dataclass(frozen=True)
class LoadLoaderConfig:
    zip_path: Path
    task_id: int = 1  # Task 1 contains the full historical timeline in a single file
    drop_na_target: bool = True
    interpolate_missing_timestamps: bool = True
    interpolate_columns: Optional[List[str]] = None  # default: all numeric columns
    freq: str = "h"


def load_gefcom_load_task(config: LoadLoaderConfig) -> pd.DataFrame:
    """
    Returns a MultiIndex dataframe with index levels [ZONEID, timestamp].
    Columns include: LOAD, w1..w25.
    """
    task = int(config.task_id)
    internal = f"Load/Task {task}/L{task}-train.csv"

    df = read_csv_from_root(config.zip_path, internal)

    if "ZONEID" not in df.columns or "TIMESTAMP" not in df.columns:
        raise ValueError(f"Unexpected load schema in {internal}: columns={df.columns.tolist()}")

    parse_res = parse_gefcom_load_timestamps(df["TIMESTAMP"].astype(str).tolist())
    df = df.drop(columns=["TIMESTAMP"])
    df.insert(1, "timestamp", parse_res.timestamps)

    df = df.sort_values(["ZONEID", "timestamp"])

    if config.drop_na_target and "LOAD" in df.columns:
        df = df[df["LOAD"].notnull()].copy()

    df = df.set_index(["ZONEID", "timestamp"]).sort_index()

    if config.interpolate_missing_timestamps:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        interp_cols = config.interpolate_columns or numeric_cols

        out = []
        for zone, g in df.groupby(level=0, sort=False):
            g = g.droplevel(0).sort_index()
            if not g.index.is_unique:
                # Some Task 1 timestamps can parse ambiguously and collide after continuity parsing.
                # Reindex() requires a unique index, so we deduplicate by aggregating duplicates.
                dup_count = int(g.index.duplicated(keep=False).sum())
                warnings.warn(
                    f"Duplicate timestamps detected for ZONEID={zone} (rows={dup_count}). "
                    "Aggregating duplicates by mean before reindexing.",
                    RuntimeWarning,
                )
                g = g.groupby(level=0).mean(numeric_only=True).sort_index()
            full_idx = pd.date_range(g.index.min(), g.index.max(), freq=config.freq)
            g2 = g.reindex(full_idx)

            cols_to_interp = [c for c in interp_cols if c in g2.columns]
            if cols_to_interp:
                g2[cols_to_interp] = g2[cols_to_interp].interpolate(method="time", limit_direction="both")

            g2["ZONEID"] = zone
            g2 = g2.reset_index(names="timestamp").set_index(["ZONEID", "timestamp"])
            out.append(g2)

        df = pd.concat(out).sort_index()

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load GEFCom2014 load track and build a continuous hourly dataframe."
    )
    parser.add_argument("--zip_path", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--out_parquet", type=str, default="")
    args = parser.parse_args()

    cfg = LoadLoaderConfig(zip_path=Path(args.zip_path), task_id=args.task_id)
    df = load_gefcom_load_task(cfg)
    print(df.head())
    print(df.tail())
    print("Shape:", df.shape)

    if args.out_parquet:
        df.to_parquet(args.out_parquet)
        print("Saved:", args.out_parquet)

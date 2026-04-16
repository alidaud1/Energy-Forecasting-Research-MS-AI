from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import yaml

# Allow running as:
# - python -m src.prepare_data   (recommended)
# - python src/prepare_data.py   (common on Windows)
if __package__ is None or __package__ == "":  # pragma: no cover
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.data_loaders.load_track_loader import LoadLoaderConfig, load_gefcom_load_task
from src.data_loaders.scaling import LeakageSafeScaler
from src.data_loaders.solar_features import add_time_features
from src.data_loaders.solar_track_loader import SolarLoaderConfig, load_gefcom_solar_task
from src.data_loaders.splits import chronological_split_by_zone

LOGGER = logging.getLogger("prepare_data")


TIME_FEATURE_COLS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


@dataclass
class PrepConfig:
    # Inputs
    load_zip: str = "data/raw/GEFCom2014-L_V2.zip"
    solar_zip: str = "data/raw/GEFCom2014-S_V2.zip"
    load_task_id: int = 1
    solar_task_id: int = 15

    # Preprocessing
    add_time_features: bool = True
    solar_decumulate: bool = True

    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20

    # Scaling
    scaler_kind: str = "standard"  # {"standard","minmax"}

    # Feature policy
    load_future_include_weather: bool = False

    # Outputs
    out_dir: str = "data/processed"

    # Manifest
    manifest_name: str = "dataset_manifest.json"


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _split_range(df: pd.DataFrame) -> Dict[str, str]:
    ts = df.index.get_level_values("timestamp")
    return {"start": pd.Timestamp(ts.min()).isoformat(), "end": pd.Timestamp(ts.max()).isoformat()}


def _split_range_by_zone(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for zone, g in df.groupby(level=0, sort=False):
        g = g.droplevel(0)
        out[str(zone)] = {
            "start": pd.Timestamp(g.index.min()).isoformat(),
            "end": pd.Timestamp(g.index.max()).isoformat(),
        }
    return out


def _infer_cols_load(
    df: pd.DataFrame, include_weather_future: bool
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns: feature_cols, target_cols, hist_cols, future_cols
    """
    target_cols = ["LOAD"]
    weather_cols = [c for c in df.columns if c.startswith("w")]
    time_cols = [c for c in TIME_FEATURE_COLS if c in df.columns]

    feature_cols = weather_cols + time_cols
    hist_cols = ["LOAD"] + weather_cols + time_cols

    if include_weather_future:
        future_cols = weather_cols + time_cols
    else:
        future_cols = time_cols

    return feature_cols, target_cols, hist_cols, future_cols


def _infer_cols_solar(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns: feature_cols, target_cols, hist_cols, future_cols
    """
    target_cols = ["POWER"]
    var_cols = [c for c in df.columns if c.startswith("VAR")]
    time_cols = [c for c in TIME_FEATURE_COLS if c in df.columns]

    feature_cols = var_cols + time_cols
    hist_cols = ["POWER"] + var_cols + time_cols
    future_cols = var_cols + time_cols
    return feature_cols, target_cols, hist_cols, future_cols


def _extract_scaler_stats(scaler_wrapper: LeakageSafeScaler) -> Dict[str, Any]:
    def stats_for(cols: List[str], sk_scaler: Any) -> Dict[str, Any]:
        if hasattr(sk_scaler, "mean_") and hasattr(sk_scaler, "scale_"):
            return {
                "type": "StandardScaler",
                "columns": cols,
                "mean": {c: float(m) for c, m in zip(cols, sk_scaler.mean_)},
                "std": {c: float(s) for c, s in zip(cols, sk_scaler.scale_)},
            }
        if hasattr(sk_scaler, "data_min_") and hasattr(sk_scaler, "data_max_"):
            return {
                "type": "MinMaxScaler",
                "columns": cols,
                "data_min": {c: float(mn) for c, mn in zip(cols, sk_scaler.data_min_)},
                "data_max": {c: float(mx) for c, mx in zip(cols, sk_scaler.data_max_)},
            }
        return {"type": type(sk_scaler).__name__, "columns": cols}

    x_scaler = getattr(scaler_wrapper, "_x_scaler", None)
    y_scaler = getattr(scaler_wrapper, "_y_scaler", None)
    return {
        "kind": scaler_wrapper.kind,
        "features": stats_for(scaler_wrapper.feature_cols, x_scaler),
        "targets": stats_for(scaler_wrapper.target_cols, y_scaler),
    }


def _save_parquets(
    track_dir: Path,
    df_clean: pd.DataFrame,
    splits_raw: Dict[str, pd.DataFrame],
    splits_scaled: Dict[str, pd.DataFrame],
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    _ensure_dir(track_dir)

    clean_path = track_dir / "clean.parquet"
    df_clean.to_parquet(clean_path)
    paths["clean"] = str(clean_path.as_posix())

    for k, v in splits_raw.items():
        p = track_dir / f"{k}.parquet"
        v.to_parquet(p)
        paths[k] = str(p.as_posix())

    for k, v in splits_scaled.items():
        p = track_dir / f"{k}_scaled.parquet"
        v.to_parquet(p)
        paths[f"{k}_scaled"] = str(p.as_posix())

    return paths


def _prepare_one_track(
    name: str,
    df_clean: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    hist_cols: List[str],
    future_cols: List[str],
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    scaler_kind: str,
) -> Dict[str, Any]:
    split = chronological_split_by_zone(
        df_clean,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    splits_raw = {"train": split.train, "val": split.val, "test": split.test}

    scaler = LeakageSafeScaler(feature_cols=feature_cols, target_cols=target_cols, kind=scaler_kind).fit(split.train)
    splits_scaled = {k: scaler.transform(v) for k, v in splits_raw.items()}

    track_dir = out_dir / name
    parquet_paths = _save_parquets(track_dir, df_clean, splits_raw, splits_scaled)

    scalers_dir = _ensure_dir(out_dir / "scalers")
    scaler_path = scalers_dir / f"{name}_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    return {
        "name": name,
        "paths": {**parquet_paths, "scaler": str(scaler_path.as_posix())},
        "columns": {
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "hist_cols": hist_cols,
            "future_cols": future_cols,
        },
        "splits": {
            "global_ranges": {k: _split_range(v) for k, v in splits_raw.items()},
            "per_zone_ranges": {k: _split_range_by_zone(v) for k, v in splits_raw.items()},
        },
        "scaler_stats": _extract_scaler_stats(scaler),
        "dataset_stats": {
            "n_rows_clean": int(len(df_clean)),
            "n_rows_train": int(len(split.train)),
            "n_rows_val": int(len(split.val)),
            "n_rows_test": int(len(split.test)),
            "n_zones": int(df_clean.index.get_level_values("ZONEID").nunique()),
            "columns_clean": list(df_clean.columns),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-A data preparation: Load + Solar tracks -> parquet + scalers + dataset_manifest.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="", help="YAML config path (optional).")

    p.add_argument("--load_zip", type=str, default=None)
    p.add_argument("--solar_zip", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--load_task_id", type=int, default=None)
    p.add_argument("--solar_task_id", type=int, default=None)

    g1 = p.add_mutually_exclusive_group()
    g1.add_argument("--add_time_features", dest="add_time_features", action="store_true")
    g1.add_argument("--no_add_time_features", dest="add_time_features", action="store_false")
    p.set_defaults(add_time_features=None)

    g2 = p.add_mutually_exclusive_group()
    g2.add_argument("--solar_decumulate", dest="solar_decumulate", action="store_true")
    g2.add_argument("--no_solar_decumulate", dest="solar_decumulate", action="store_false")
    p.set_defaults(solar_decumulate=None)

    g3 = p.add_mutually_exclusive_group()
    g3.add_argument("--load_future_include_weather", dest="load_future_include_weather", action="store_true")
    g3.add_argument("--load_future_time_only", dest="load_future_include_weather", action="store_false")
    p.set_defaults(load_future_include_weather=None)

    p.add_argument("--train_ratio", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--scaler_kind", type=str, default=None, choices=["standard", "minmax"])

    return p


def _finalize_config(args: argparse.Namespace) -> PrepConfig:
    base = PrepConfig().__dict__
    yaml_cfg = _load_yaml_config(args.config)
    merged = _deep_update(base, yaml_cfg)

    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is not None:
            merged[key] = val

    return PrepConfig(**merged)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = build_arg_parser().parse_args()
    cfg = _finalize_config(args)

    out_dir = _ensure_dir(Path(cfg.out_dir))
    LOGGER.info("Starting Phase-A prepare_data with config: %s", cfg)

    # LOAD
    load_df = load_gefcom_load_task(LoadLoaderConfig(zip_path=Path(cfg.load_zip), task_id=cfg.load_task_id))
    if cfg.add_time_features:
        load_df = add_time_features(load_df)
    load_feature_cols, load_target_cols, load_hist_cols, load_future_cols = _infer_cols_load(
        load_df, include_weather_future=cfg.load_future_include_weather
    )

    # SOLAR
    solar_df = load_gefcom_solar_task(
        SolarLoaderConfig(
            zip_path=Path(cfg.solar_zip),
            task_id=cfg.solar_task_id,
            apply_decumulation=cfg.solar_decumulate,
        )
    )
    if cfg.add_time_features:
        solar_df = add_time_features(solar_df)
    solar_feature_cols, solar_target_cols, solar_hist_cols, solar_future_cols = _infer_cols_solar(solar_df)

    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": {"name": "phase_a_prepare_data", "version": "1.0.0"},
        "system": {"python": platform.python_version(), "platform": platform.platform()},
        "inputs": {
            "load_zip": cfg.load_zip,
            "solar_zip": cfg.solar_zip,
            "load_task_id": cfg.load_task_id,
            "solar_task_id": cfg.solar_task_id,
        },
        "preprocessing": {
            "add_time_features": cfg.add_time_features,
            "solar_decumulate": cfg.solar_decumulate,
            "load_future_include_weather": cfg.load_future_include_weather,
            "time_feature_cols": TIME_FEATURE_COLS,
        },
        "splits": {"train_ratio": cfg.train_ratio, "val_ratio": cfg.val_ratio, "test_ratio": cfg.test_ratio},
        "scaling": {"scaler_kind": cfg.scaler_kind},
        "tracks": {},
    }

    manifest["tracks"]["load"] = _prepare_one_track(
        name="load",
        df_clean=load_df,
        feature_cols=load_feature_cols,
        target_cols=load_target_cols,
        hist_cols=load_hist_cols,
        future_cols=load_future_cols,
        out_dir=out_dir,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        scaler_kind=cfg.scaler_kind,
    )

    manifest["tracks"]["solar"] = _prepare_one_track(
        name="solar",
        df_clean=solar_df,
        feature_cols=solar_feature_cols,
        target_cols=solar_target_cols,
        hist_cols=solar_hist_cols,
        future_cols=solar_future_cols,
        out_dir=out_dir,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        scaler_kind=cfg.scaler_kind,
    )

    manifest_path = out_dir / cfg.manifest_name
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    LOGGER.info("Wrote manifest: %s", manifest_path)
    LOGGER.info("Phase A complete: parquet + scalers + manifest are ready in %s", out_dir)


if __name__ == "__main__":
    main()


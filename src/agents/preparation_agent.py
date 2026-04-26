from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
import joblib

from src.data_loaders.scaling import LeakageSafeScaler

from .base_agent import BaseAgent


class PreparationAgent(BaseAgent):
    """
    Verifies Phase A artifacts from dataset_manifest.json and signals readiness.
    """

    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        # Phase 103: listen to optional system_directive so we can record human preferences
        # about preparation/cleaning/loss without breaking the non-interactive pipeline.
        self.subscribe(topics=["prepare", "system_directive"], start_from="latest")
        self._prep_preferences: Dict[str, Any] = {}

    def _resolve_path(self, manifest_path: Path, artifact_path: str) -> Path:
        p = Path(artifact_path)
        if p.is_absolute():
            return p
        # Try resolving relative to the manifest directory first.
        cand1 = (manifest_path.parent / p).resolve()
        if cand1.exists():
            return cand1

        # Fallback: resolve relative to the current working directory (project root in typical runs).
        cand2 = (Path.cwd() / p).resolve()
        if cand2.exists():
            return cand2

        # Last resort: if the path was written as "data/processed/..." and we are already in that folder,
        # avoid doubling "data/processed".
        if str(p).replace("\\", "/").startswith("data/processed/"):
            trimmed = str(p).replace("\\", "/").split("data/processed/", 1)[1]
            cand3 = (Path.cwd() / "data" / "processed" / trimmed).resolve()
            return cand3

        return cand2

    def _verify_manifest_and_paths(self, manifest_path: str, track: str) -> Dict[str, Any]:
        mp = Path(manifest_path)
        if not mp.exists():
            raise FileNotFoundError(f"dataset_manifest.json not found: {mp}")

        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if "tracks" not in manifest or track not in manifest["tracks"]:
            raise ValueError(f"Track '{track}' not found in manifest['tracks'].")

        t = manifest["tracks"][track]
        paths = t["paths"]

        required = ["train_scaled", "val_scaled", "test_scaled", "scaler"]
        missing = [k for k in required if k not in paths]
        if missing:
            raise ValueError(f"Manifest missing required paths for track '{track}': {missing}")

        resolved_paths = {k: str(self._resolve_path(mp, paths[k])) for k in required}

        for k, p_str in resolved_paths.items():
            p = Path(p_str)
            if not p.exists():
                raise FileNotFoundError(f"Missing artifact for '{track}': {k} -> {p}")

        # Lightweight parquet sanity check (must be readable, columns exist)
        df_head = pd.read_parquet(resolved_paths["train_scaled"]).head(5)

        cols = t.get("columns", {})
        for must in ["hist_cols", "future_cols", "target_cols"]:
            if must not in cols or not cols[must]:
                raise ValueError(f"Manifest columns missing '{must}' for track '{track}'.")

        # Return full paths for downstream agents so everything is unambiguous.
        full_paths = dict(paths)
        for k, v in resolved_paths.items():
            full_paths[k] = v

        return {
            "manifest_path": str(mp.resolve()),
            "track": track,
            "paths": full_paths,
            "columns": cols,
            "train_head_columns": list(df_head.columns),
        }

    def _summarize_split(self, path: str, *, target_col: str) -> Dict[str, Any]:
        """
        Production-friendly summary for interactive prep gating.
        Reads only the target column to reduce IO.
        """
        df = pd.read_parquet(path, columns=[target_col])
        zones = df.index.get_level_values(0).unique()
        ts = df.index.get_level_values(1)
        return {
            "path": str(path),
            "n_rows": int(df.shape[0]),
            "n_zones": int(len(zones)),
            "ts_min": str(pd.Timestamp(ts.min())),
            "ts_max": str(pd.Timestamp(ts.max())),
            "na_target": int(df[target_col].isna().sum()),
        }

    def _select_top_by_corr(self, df: pd.DataFrame, *, target_col: str, candidates: list[str], top_ratio: float) -> list[str]:
        r = float(top_ratio)
        if r <= 0 or not candidates:
            return []
        if r >= 1:
            return list(candidates)
        scores = []
        y = df[target_col]
        for c in candidates:
            try:
                s = df[c]
                v = float(s.corr(y))
                if v == v:
                    scores.append((abs(v), c))
            except Exception:
                continue
        scores.sort(reverse=True)
        k = max(0, int(np.ceil(len(scores) * r)))
        return [c for _v, c in scores[:k]]

    def _add_calendar_features(self, df: pd.DataFrame, *, mode: str) -> tuple[pd.DataFrame, list[str]]:
        """
        Table III: calendar {none, numerical, categorical, trigonometric}.
        Adds columns and returns (df, calendar_cols).
        """
        m = str(mode or "none").lower()
        ts = df.index.get_level_values(1)
        cal = pd.DataFrame(index=df.index)
        if m == "none":
            return df, []
        if m == "numerical":
            cal["hour"] = ts.hour.astype(int)
            cal["dow"] = ts.dayofweek.astype(int)
            cal["month"] = ts.month.astype(int)
            out = df.drop(columns=[c for c in ["hour", "dow", "month"] if c in df.columns], errors="ignore")
            out = out.drop(columns=[c for c in df.columns if c.endswith("_sin") or c.endswith("_cos")], errors="ignore")
            out = out.join(cal)
            return out, ["hour", "dow", "month"]
        if m == "categorical":
            cal["hour"] = ts.hour.astype(int)
            cal["dow"] = ts.dayofweek.astype(int)
            cal["month"] = ts.month.astype(int)
            d = pd.get_dummies(cal, columns=["hour", "dow", "month"], prefix=["hour", "dow", "month"], drop_first=True)
            out = df.drop(columns=[c for c in df.columns if c.startswith(("hour_", "dow_", "month_"))], errors="ignore")
            out = out.drop(columns=[c for c in ["hour", "dow", "month"] if c in out.columns], errors="ignore")
            out = out.drop(columns=[c for c in out.columns if c.endswith("_sin") or c.endswith("_cos")], errors="ignore")
            out = out.join(d)
            return out, list(d.columns)
        if m == "trigonometric":
            hour = ts.hour.to_numpy()
            dow = ts.dayofweek.to_numpy()
            month = ts.month.to_numpy()
            cal["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            cal["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            cal["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            cal["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
            cal["month_sin"] = np.sin(2 * np.pi * month / 12.0)
            cal["month_cos"] = np.cos(2 * np.pi * month / 12.0)
            out = df.drop(columns=[c for c in cal.columns if c in df.columns], errors="ignore").join(cal)
            return out, list(cal.columns)
        raise ValueError(f"Unsupported calendar mode: {mode}")

    def _add_group_lags(self, df: pd.DataFrame, *, src_col: str, lags: list[int], prefix: str) -> tuple[pd.DataFrame, list[str]]:
        out = df
        new_cols: list[str] = []
        for lag in lags:
            name = f"{prefix}_lag{int(lag)}"
            out[name] = out.groupby(level=0, sort=False)[src_col].shift(int(lag))
            new_cols.append(name)
        return out, new_cols

    def _apply_feature_search(
        self,
        *,
        manifest_path: Path,
        track: str,
        base_paths: dict,
        base_columns: dict,
        feature_search: dict,
    ) -> tuple[dict, dict]:
        """
        Table III feature-group construction/search.
        Produces run-scoped parquets under logs/runs/<run_id>/prepared/<track>/ and returns updated (paths, columns).
        """
        # Require unscaled splits for feature construction.
        for k in ("train", "val", "test"):
            if k not in base_paths:
                raise ValueError(f"Feature search requires unscaled split path '{k}' in manifest paths for track '{track}'.")

        p_train = str(self._resolve_path(manifest_path, base_paths["train"]))
        p_val = str(self._resolve_path(manifest_path, base_paths["val"]))
        p_test = str(self._resolve_path(manifest_path, base_paths["test"]))

        df_train = pd.read_parquet(p_train)
        df_val = pd.read_parquet(p_val)
        df_test = pd.read_parquet(p_test)

        target_col = (base_columns.get("target_cols") or [""])[0]
        if not target_col or target_col not in df_train.columns:
            raise ValueError(f"Invalid target column for feature search: {target_col}")

        cfg = dict(feature_search or {})
        calendar_mode = str(cfg.get("calendar", "trigonometric"))
        other_cfg = dict(cfg.get("other_features") or {"mode": "all"})
        load_lag_cfg = dict(cfg.get("load_lags") or {"mode": "none"})
        hist_seq_cfg = dict(cfg.get("historical_sequence") or {"frequency": 1, "length": 1})
        temp_cfg = dict(cfg.get("temperature_lags") or {"mode": "none"})
        interaction_mode = str(cfg.get("calendar_temperature_interaction", "none"))

        # 1) Calendar features
        df_train, cal_cols = self._add_calendar_features(df_train, mode=calendar_mode)
        df_val, _ = self._add_calendar_features(df_val, mode=calendar_mode)
        df_test, _ = self._add_calendar_features(df_test, mode=calendar_mode)

        # Candidate exogenous columns (track-specific heuristic)
        base_exog = [c for c in df_train.columns if c != target_col and c not in cal_cols]
        if track == "load":
            exog_candidates = [c for c in base_exog if c.startswith("w")]
        elif track == "solar":
            exog_candidates = [c for c in base_exog if c.startswith("VAR")]
        else:
            exog_candidates = list(base_exog)

        # 2) Temperature lags (optional; requires temp_col)
        temp_lag_cols: list[str] = []
        if str(temp_cfg.get("mode", "none")).lower() == "correlation":
            temp_col = str(temp_cfg.get("temp_col") or "")
            if not temp_col or temp_col not in df_train.columns:
                raise ValueError("temperature_lags.mode=correlation requires feature_search.temperature_lags.temp_col present in the dataset.")
            possible_lags = list(range(1, 169))
            scores = []
            for lag in possible_lags:
                s = df_train.groupby(level=0, sort=False)[temp_col].shift(lag)
                v = float(s.corr(df_train[target_col]))
                if v == v:
                    scores.append((abs(v), lag))
            scores.sort(reverse=True)
            top_ratio = float(temp_cfg.get("top_ratio", 0.0))
            k = max(0, int(np.ceil(len(scores) * max(0.0, min(1.0, top_ratio)))))
            chosen = [lag for _v, lag in scores[:k]]
            for lag in chosen:
                df_train, newc = self._add_group_lags(df_train, src_col=temp_col, lags=[lag], prefix=f"{temp_col}")
                df_val, _ = self._add_group_lags(df_val, src_col=temp_col, lags=[lag], prefix=f"{temp_col}")
                df_test, _ = self._add_group_lags(df_test, src_col=temp_col, lags=[lag], prefix=f"{temp_col}")
                temp_lag_cols.extend(newc)

        # 3) Load lag values {none, correlation, fixed}
        load_lag_cols: list[str] = []
        load_mode = str(load_lag_cfg.get("mode", "none")).lower()
        if load_mode == "fixed":
            fixed_lags = load_lag_cfg.get("fixed_lags") or [1, 2, 3, 24, 48, 72, 168]
            lags = [int(x) for x in fixed_lags]
            df_train, newc = self._add_group_lags(df_train, src_col=target_col, lags=lags, prefix=target_col)
            df_val, _ = self._add_group_lags(df_val, src_col=target_col, lags=lags, prefix=target_col)
            df_test, _ = self._add_group_lags(df_test, src_col=target_col, lags=lags, prefix=target_col)
            load_lag_cols.extend(newc)
        elif load_mode == "correlation":
            possible_lags = list(range(1, 169))
            scores = []
            for lag in possible_lags:
                s = df_train.groupby(level=0, sort=False)[target_col].shift(lag)
                v = float(s.corr(df_train[target_col]))
                if v == v:
                    scores.append((abs(v), lag))
            scores.sort(reverse=True)
            top_ratio = float(load_lag_cfg.get("top_ratio", 0.0))
            k = max(0, int(np.ceil(len(scores) * max(0.0, min(1.0, top_ratio)))))
            chosen = [lag for _v, lag in scores[:k]]
            if chosen:
                df_train, newc = self._add_group_lags(df_train, src_col=target_col, lags=chosen, prefix=target_col)
                df_val, _ = self._add_group_lags(df_val, src_col=target_col, lags=chosen, prefix=target_col)
                df_test, _ = self._add_group_lags(df_test, src_col=target_col, lags=chosen, prefix=target_col)
                load_lag_cols.extend(newc)

        # 4) Calendar-Temperature interaction {none, all}
        interaction_cols: list[str] = []
        if str(interaction_mode).lower() == "all" and cal_cols and temp_lag_cols:
            for cal_col in cal_cols:
                for lag_col in temp_lag_cols:
                    name = f"{cal_col}_x_{lag_col}"
                    df_train[name] = df_train[cal_col] * df_train[lag_col]
                    df_val[name] = df_val[cal_col] * df_val[lag_col]
                    df_test[name] = df_test[cal_col] * df_test[lag_col]
                    interaction_cols.append(name)

        # 5) Historical load sequence: frequency + length
        seq_cols: list[str] = []
        try:
            freq = int(hist_seq_cfg.get("frequency", 1))
            length = int(hist_seq_cfg.get("length", 1))
        except Exception:
            freq, length = 1, 1
        freq_allowed = {1, 2, 3, 4, 6, 12, 24}
        if freq not in freq_allowed:
            raise ValueError(f"historical_sequence.frequency must be one of {sorted(freq_allowed)}")
        if not (1 <= length <= 24):
            raise ValueError("historical_sequence.length must be in [1, 24]")
        lags = [freq * k for k in range(1, length + 1)]
        df_train, newc = self._add_group_lags(df_train, src_col=target_col, lags=lags, prefix=f"hist_{target_col}")
        df_val, _ = self._add_group_lags(df_val, src_col=target_col, lags=lags, prefix=f"hist_{target_col}")
        df_test, _ = self._add_group_lags(df_test, src_col=target_col, lags=lags, prefix=f"hist_{target_col}")
        seq_cols.extend(newc)

        # 6) Other features {none, correlation}
        other_mode = str(other_cfg.get("mode", "all")).lower()
        if other_mode == "none":
            exog_selected: list[str] = []
        elif other_mode == "correlation":
            top_ratio = float(other_cfg.get("top_ratio", 1.0))
            exog_selected = self._select_top_by_corr(df_train, target_col=target_col, candidates=exog_candidates, top_ratio=top_ratio)
        else:
            exog_selected = list(exog_candidates)

        # Assemble final feature set for scaling/modeling (exclude target).
        feature_cols = sorted(set(cal_cols + exog_selected + temp_lag_cols + load_lag_cols + interaction_cols + seq_cols))
        # Impute NaNs introduced by shifting for feature columns only.
        for df in (df_train, df_val, df_test):
            if feature_cols:
                df[feature_cols] = df[feature_cols].astype(np.float32)
                df[feature_cols] = df[feature_cols].fillna(0.0)

        # Fit scaler on train only, then transform all splits.
        scaler = LeakageSafeScaler(feature_cols=feature_cols, target_cols=[target_col], kind="standard").fit(df_train)
        train_scaled = scaler.transform(df_train)
        val_scaled = scaler.transform(df_val)
        test_scaled = scaler.transform(df_test)

        out_root = (Path.cwd() / "logs" / "runs" / self.run_id / "prepared" / track).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "scalers").mkdir(parents=True, exist_ok=True)

        p_train_out = out_root / "train.parquet"
        p_val_out = out_root / "val.parquet"
        p_test_out = out_root / "test.parquet"
        p_train_s = out_root / "train_scaled.parquet"
        p_val_s = out_root / "val_scaled.parquet"
        p_test_s = out_root / "test_scaled.parquet"
        p_scaler = out_root / "scalers" / f"{track}_scaler.joblib"

        df_train.to_parquet(p_train_out.as_posix())
        df_val.to_parquet(p_val_out.as_posix())
        df_test.to_parquet(p_test_out.as_posix())
        train_scaled.to_parquet(p_train_s.as_posix())
        val_scaled.to_parquet(p_val_s.as_posix())
        test_scaled.to_parquet(p_test_s.as_posix())
        joblib.dump(scaler, p_scaler.as_posix())

        new_paths = dict(base_paths)
        new_paths.update(
            {
                "train": p_train_out.as_posix(),
                "val": p_val_out.as_posix(),
                "test": p_test_out.as_posix(),
                "train_scaled": p_train_s.as_posix(),
                "val_scaled": p_val_s.as_posix(),
                "test_scaled": p_test_s.as_posix(),
                "scaler": p_scaler.as_posix(),
            }
        )
        new_cols = {
            "feature_cols": feature_cols,
            "target_cols": [target_col],
            "hist_cols": [target_col] + feature_cols,
            "future_cols": list(cal_cols),
        }
        return new_paths, new_cols

    def tick(self) -> None:
        for msg in self.poll(max_messages=50):
            # Record any preparation-related preferences (best-effort, non-blocking).
            if msg.kind == "event" and msg.topic.startswith(f"system_directive/{self.run_id}"):
                p = msg.payload or {}
                overrides = p.get("overrides") or {}
                if isinstance(overrides, dict):
                    # Keep everything; downstream can filter. This is telemetry only.
                    self._prep_preferences.update({str(k): overrides[k] for k in overrides.keys()})
                continue

            if msg.kind != "command":
                continue
            if msg.payload.get("command") != "prepare":
                continue

            corr_id = msg.correlation_id
            run_cfg = msg.payload.get("run_config", {}) or {}

            try:
                # Phase 103: explicit staged telemetry for Fig.4-like preparation workflow.
                self.publish(
                    topic_type="prepare_step",
                    kind="event",
                    payload={"event": "metadata_collected", "manifest_path": msg.payload.get("manifest_path"), "track": msg.payload.get("track")},
                    correlation_id=corr_id,
                )
                info = self._verify_manifest_and_paths(
                    manifest_path=msg.payload["manifest_path"],
                    track=msg.payload["track"],
                )
                self.publish(
                    topic_type="prepare_step",
                    kind="event",
                    payload={"event": "validity_check_passed", "track": info["track"]},
                    correlation_id=corr_id,
                )

                cols = info["columns"]
                target_col = (cols.get("target_cols") or [""])[0]
                self.publish(
                    topic_type="prepare_step",
                    kind="event",
                    payload={"event": "columns_defined", "track": info["track"], "target_col": target_col, "n_hist": len(cols.get("hist_cols") or []), "n_future": len(cols.get("future_cols") or [])},
                    correlation_id=corr_id,
                )
                # Optional Table III feature search: if provided, generate run-scoped parquets/scaler
                # and update the paths/columns we announce as "ready". Defaults keep current behavior.
                feature_search = (run_cfg.get("feature_search") or {}) if isinstance(run_cfg, dict) else {}
                if isinstance(feature_search, dict) and feature_search:
                    new_paths, new_cols = self._apply_feature_search(
                        manifest_path=Path(info["manifest_path"]),
                        track=info["track"],
                        base_paths=info["paths"],
                        base_columns=info["columns"],
                        feature_search=feature_search,
                    )
                    info["paths"] = new_paths
                    info["columns"] = new_cols
                    cols = new_cols
                    target_col = (cols.get("target_cols") or [""])[0]

                summary = {}
                for split in ("train_scaled", "val_scaled", "test_scaled"):
                    p = info["paths"].get(split)
                    if p and target_col:
                        summary[split] = self._summarize_split(p, target_col=target_col)
                self.publish(
                    topic_type="prepare_step",
                    kind="event",
                    payload={"event": "cleaning_preferences_captured", "track": info["track"], "preferences": dict(self._prep_preferences)},
                    correlation_id=corr_id,
                )
                self.publish(
                    topic_type="prepare",
                    kind="event",
                    payload={
                        "event": "ready",
                        "track": info["track"],
                        "manifest_path": info["manifest_path"],
                        "paths": info["paths"],
                        "columns": info["columns"],
                        "summary": summary,
                        "feature_search": feature_search if isinstance(feature_search, dict) else {},
                        "prep_preferences": dict(self._prep_preferences),
                        "run_config": run_cfg,
                    },
                    correlation_id=corr_id,
                )
            except Exception as e:
                self.publish(
                    topic_type="prepare",
                    kind="error",
                    payload={"event": "not_ready", "error": str(e)},
                    correlation_id=corr_id,
                    level="error",
                )

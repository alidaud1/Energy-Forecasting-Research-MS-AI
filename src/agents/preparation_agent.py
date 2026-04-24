from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

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

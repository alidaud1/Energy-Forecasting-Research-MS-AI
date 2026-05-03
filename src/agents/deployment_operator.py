from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_agent import BaseAgent


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass(frozen=True)
class DeploymentConfig:
    """
    Minimal deployment-stage adjustment configuration.
    """

    message_pool_path: str = "logs/message_pool.jsonl"


class DeploymentOperatorAgent(BaseAgent):
    """
    Phase 105: Deployment Operator

    Responsibility:
    - Apply postprocess adjustment rules (Table I) after a model produces initial forecasts.
    - Support:
      - Manual/time-based scaling or bias (hours mask)
      - Load-based trigger: apply only if y_pred >= theta
      - External-variable trigger: apply only if phi_col >= phi_theta
    - Emit traceable artifacts:
      - adjusted_predictions_path (parquet) containing both original and adjusted predictions.
    - Publish `deploy:result` with original vs adjusted metrics and the applied mask stats.
    """

    def __init__(self, cfg, pool, dcfg: Optional[DeploymentConfig] = None):
        super().__init__(cfg, pool)
        self.dcfg = dcfg or DeploymentConfig()
        self.subscribe(topics=["deploy"], start_from="latest")

        self.run_dir = Path("logs") / "runs" / self.run_id
        (self.run_dir / "child_runs").mkdir(parents=True, exist_ok=True)

    def _hours_mask(self, ts: pd.Series, hours_spec: str) -> np.ndarray:
        """
        hours_spec: "18-21" or "18,19,20"
        """
        if not hours_spec:
            return np.ones(len(ts), dtype=bool)
        hs = str(hours_spec).strip()
        hours: set[int] = set()
        if "-" in hs:
            a, b = hs.split("-", 1)
            try:
                start = int(a)
                end = int(b)
                for h in range(start, end + 1):
                    hours.add(h)
            except Exception:
                return np.ones(len(ts), dtype=bool)
        else:
            for p in hs.split(","):
                try:
                    hours.add(int(p.strip()))
                except Exception:
                    continue
        t = pd.to_datetime(ts)
        return t.dt.hour.isin(sorted(hours)).to_numpy()

    def _load_phi_series(self, *, data_path: str, phi_col: str) -> pd.DataFrame:
        """
        Load phi_t (external variable) from an unscaled parquet split and return a small DF with keys:
          - timestamp
          - ZONEID (if present in index)
          - phi_col
        """
        df = pd.read_parquet(data_path, columns=[phi_col])
        # Expect multi-index [ZONEID, timestamp]
        if list(df.index.names) == ["ZONEID", "timestamp"]:
            out = df.reset_index()
            out["timestamp"] = pd.to_datetime(out["timestamp"])
            return out[["ZONEID", "timestamp", phi_col]]
        # Fallback: if timestamp is a column
        if "timestamp" in df.columns:
            out = df.copy()
            out["timestamp"] = pd.to_datetime(out["timestamp"])
            if "ZONEID" not in out.columns:
                out["ZONEID"] = 0
            return out[["ZONEID", "timestamp", phi_col]]
        raise ValueError(f"Unsupported data format for phi_col merge: index={df.index.names}, columns={list(df.columns)}")

    def _metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        yt = df["y_true"].to_numpy(dtype=float)
        yp = df["y_pred"].to_numpy(dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp)
        if not m.any():
            return {"mae": float("nan"), "rmse": float("nan")}
        mae = float(np.mean(np.abs(yt[m] - yp[m])))
        rmse = _rmse_1d(yt[m], yp[m])
        return {"mae": mae, "rmse": rmse}

    def _apply_adjustment(self, df: pd.DataFrame, adj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns:
          {df_adj, mask_applied, stats}
        """
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"])
        out["y_pred_original"] = out["y_pred"]

        mask = np.ones(len(out), dtype=bool)

        # Time-based mask
        hours = str(adj.get("hours") or "")
        if hours:
            mask &= self._hours_mask(out["timestamp"], hours)

        # Load-based trigger: y_pred >= theta
        theta = adj.get("theta", None)
        if theta is not None and str(theta) != "":
            try:
                th = float(theta)
                mask &= out["y_pred_original"].to_numpy(dtype=float) >= th
            except Exception:
                pass

        # External-variable trigger: phi_col >= phi_theta
        phi_col = str(adj.get("phi_col") or "").strip()
        phi_theta = adj.get("phi_theta", None)
        if phi_col and phi_theta is not None and str(phi_theta) != "":
            if phi_col in out.columns:
                try:
                    pth = float(phi_theta)
                    mask &= out[phi_col].to_numpy(dtype=float) >= pth
                except Exception:
                    pass

        action = str(adj.get("action") or "").lower()
        if action not in ("scale", "bias", "accept", "reject", ""):
            raise ValueError(f"Invalid deployment action: {action!r}. Expected scale|bias|accept|reject.")

        if action == "scale":
            try:
                factor = float(adj.get("factor"))
            except Exception as e:
                raise ValueError(f"Invalid scale factor: {adj.get('factor')!r}") from e
            if not np.isfinite(factor):
                raise ValueError(f"Invalid scale factor (not finite): {factor}")
            out.loc[mask, "y_pred"] = out.loc[mask, "y_pred"] * factor

        elif action == "bias":
            try:
                add = float(adj.get("add"))
            except Exception as e:
                raise ValueError(f"Invalid bias add value: {adj.get('add')!r}") from e
            if not np.isfinite(add):
                raise ValueError(f"Invalid bias add value (not finite): {add}")
            out.loc[mask, "y_pred"] = out.loc[mask, "y_pred"] + add
        else:
            # accept/reject/unknown -> no change
            pass

        stats = {
            "n_rows": int(len(out)),
            "n_applied": int(mask.sum()),
            "n_skipped": int((~mask).sum()),
        }
        return {"df_adj": out, "mask": mask, "stats": stats}

    def tick(self) -> None:
        for msg in self.poll(max_messages=50):
            if msg.kind != "command":
                continue
            if msg.payload.get("command") != "deploy":
                continue

            corr_id = msg.correlation_id
            p = msg.payload or {}
            track = str(p.get("track") or "")
            original_pred_path = str(p.get("original_predictions_path") or "")
            original_metrics = p.get("original_metrics", {}) or {}
            adj = p.get("adjustment", {}) or {}
            data_path = str(p.get("data_path") or "")

            try:
                df = pd.read_parquet(original_pred_path)
                if "timestamp" not in df.columns:
                    raise ValueError("Predictions parquet missing 'timestamp'")
                if "y_true" not in df.columns or "y_pred" not in df.columns:
                    raise ValueError("Predictions parquet missing y_true/y_pred")

                # Optional phi merge from processed data if requested and not already present.
                phi_col = str(adj.get("phi_col") or "").strip()
                if phi_col and phi_col not in df.columns and data_path:
                    phi_df = self._load_phi_series(data_path=data_path, phi_col=phi_col)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    if "ZONEID" in df.columns:
                        df = df.merge(phi_df, on=["ZONEID", "timestamp"], how="left")
                    else:
                        df["ZONEID"] = 0
                        df = df.merge(phi_df, on=["ZONEID", "timestamp"], how="left")

                applied = self._apply_adjustment(df, adj)
                df_adj = applied["df_adj"]
                stats = applied["stats"]

                child_id = str(p.get("child_run_id") or f"child_{_utc_compact()}_deploy")
                child_dir = self.run_dir / "child_runs" / child_id
                (child_dir / "predictions").mkdir(parents=True, exist_ok=True)
                (child_dir / "reports").mkdir(parents=True, exist_ok=True)

                df_adj["run_id"] = child_id
                df_adj["parent_run_id"] = self.run_id
                df_adj["variant"] = "adjusted"
                df_adj["adjustment"] = json.dumps(adj, ensure_ascii=False)

                out_pred = child_dir / "predictions" / f"{track}_adjusted_predictions.parquet"
                df_adj.to_parquet(out_pred)

                adj_metrics = self._metrics(df_adj)

                self.publish(
                    topic_type="deploy",
                    kind="result",
                    payload={
                        "event": "deploy_complete",
                        "track": track,
                        "child_run_id": child_id,
                        "original_predictions_path": original_pred_path,
                        "adjusted_predictions_path": out_pred.as_posix(),
                        "original_metrics": dict(original_metrics),
                        "adjusted_metrics": adj_metrics,
                        "adjustment": adj,
                        "stats": stats,
                    },
                    correlation_id=corr_id,
                )
            except Exception as e:
                self.publish(
                    topic_type="deploy",
                    kind="error",
                    payload={"event": "deploy_failed", "track": track, "error": str(e)},
                    correlation_id=corr_id,
                    level="error",
                )

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from .base_agent import BaseAgent


def _rmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass(frozen=True)
class IntegrationConfig:
    baseline_model_name: str = "xgboost"
    transformer_model_names: Tuple[str, ...] = ("patchtst", "itransformer")
    timeline_days: int = 7
    max_points_boxplot: int = 20000
    message_pool_path: str = "logs/message_pool.jsonl"


class ForecastIntegrationAgent(BaseAgent):
    """
    Phase C3 integration agent:
    - collects evaluation results for multiple models per track (baseline + transformers)
    - computes net-load when alignment exists and reports MAE/RMSE
    - generates thesis-ready plots
    - exports a lightweight system audit from message_pool.jsonl

    Backward compatibility:
    - still writes `joint_decision_report.md`
    - still emits `integrate/joint_report_ready`
    """

    POSTPROCESS_GATE_ID = "postprocess_gate_v1"

    def __init__(self, cfg, pool, icfg: Optional[IntegrationConfig] = None):
        super().__init__(cfg, pool)
        # Phase D3/D4: listen to system_directive + directive_ack/rejected for comparative audit traces.
        self.subscribe(
            topics=["evaluate", "system", "prepare", "directive", "system_directive", "directive_ack", "directive_rejected", "human_gate_decision"],
            start_from="latest",
        )

        self.icfg = icfg or IntegrationConfig()

        self.run_dir = Path("logs") / "runs" / self.run_id
        (self.run_dir / "integrated").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "child_runs").mkdir(parents=True, exist_ok=True)

        # results[track][model_name] = payload summary
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {"load": {}, "solar": {}}
        self._finalized = False
        self._interactive = False
        self._post_gate_requested = False
        self._post_done = False
        self._post_action: Dict[str, Any] = {}
        self._cached_best: Dict[str, Dict[str, Any]] = {}
        self._cached_net_info: Dict[str, Any] = {}
        self._cached_plots: Dict[str, str] = {}

    def _utc_compact(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _detect_interactive(self) -> bool:
        # Detect interactive run by looking for the TaskManager interactive system event or run_config flag.
        sys_msgs = self.pool.find_by_topic("system", topic_source=self.run_id)
        for m in sys_msgs:
            if m.kind == "event" and (m.payload or {}).get("mode") == "research_interactive":
                return True
        prep_msgs = self.pool.find_by_topic("prepare", topic_source=self.run_id)
        for m in prep_msgs:
            if m.kind == "event" and (m.payload or {}).get("event") == "ready":
                rc = (m.payload or {}).get("run_config", {}) or {}
                if bool(rc.get("interactive", False)):
                    return True
        return False

    def _postprocess_prompt(self, *, best_load: Dict[str, Any], best_solar: Dict[str, Any], net_info: Dict[str, Any]) -> str:
        lines = []
        lines.append("POST-PROCESS GATE (Phase D1)\n")
        lines.append("You can accept the forecasts or apply a simple adjustment (real, saved to child run artifacts).\n\n")
        lines.append(f"- Best load model: {best_load.get('model_name')} ({best_load.get('model_family')})\n")
        lines.append(f"- Best solar model: {best_solar.get('model_name')} ({best_solar.get('model_family')})\n")
        if net_info.get("computed"):
            m = net_info.get("metrics", {}) or {}
            lines.append(f"- Net-load: computed (MAE={m.get('mae')}, RMSE={m.get('rmse')})\n")
        else:
            lines.append(f"- Net-load: not computed ({net_info.get('reason')})\n")
        lines.append("\nCommands:\n")
        lines.append("- accept\n")
        lines.append("- scale load factor=1.05\n")
        lines.append("- scale solar factor=0.95\n")
        lines.append("- scale load hours=18-21 factor=1.05\n")
        lines.append("- bias load add=10\n")
        lines.append("- bias solar add=-0.1\n")
        return "".join(lines)

    def _parse_postprocess(self, text: str) -> Dict[str, Any]:
        raw = " ".join((text or "").strip().split())
        low = raw.lower()
        if low in ("accept", "approve", "ok", "go", "yes", "y"):
            return {"action": "accept"}
        if low in ("reject", "cancel", "stop", "no", "n"):
            return {"action": "reject"}

        parts = raw.split()
        if len(parts) < 3:
            return {"action": "unknown", "text": raw}

        action = parts[0].lower()
        target = parts[1].lower()
        kv = {}
        for p in parts[2:]:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            kv[k.strip().lower()] = v.strip()

        if action not in ("scale", "bias"):
            return {"action": "unknown", "text": raw}
        if target not in ("load", "solar"):
            return {"action": "unknown", "text": raw}

        out: Dict[str, Any] = {"action": action, "target": target}
        if "hours" in kv:
            out["hours"] = kv["hours"]
        if action == "scale":
            try:
                out["factor"] = float(kv.get("factor", ""))
            except Exception:
                return {"action": "unknown", "text": raw}
        else:
            try:
                out["add"] = float(kv.get("add", ""))
            except Exception:
                return {"action": "unknown", "text": raw}
        return out

    def _hours_mask(self, df: pd.DataFrame, hours_spec: str) -> np.ndarray:
        """
        hours_spec: "18-21" or "18,19,20"
        """
        if not hours_spec:
            return np.ones(len(df), dtype=bool)
        hs = hours_spec.strip()
        hours: set[int] = set()
        if "-" in hs:
            a, b = hs.split("-", 1)
            try:
                start = int(a)
                end = int(b)
                for h in range(start, end + 1):
                    hours.add(h)
            except Exception:
                return np.ones(len(df), dtype=bool)
        else:
            for p in hs.split(","):
                try:
                    hours.add(int(p.strip()))
                except Exception:
                    continue
        ts = pd.to_datetime(df["timestamp"])
        return ts.dt.hour.isin(sorted(hours)).to_numpy()

    def _apply_adjustment(self, df: pd.DataFrame, adj: Dict[str, Any]) -> pd.DataFrame:
        out = df.copy()
        mask = np.ones(len(out), dtype=bool)
        if "hours" in adj and adj["hours"]:
            mask = self._hours_mask(out, str(adj["hours"]))
        if adj["action"] == "scale":
            factor = float(adj["factor"])
            out.loc[mask, "y_pred"] = out.loc[mask, "y_pred"] * factor
        elif adj["action"] == "bias":
            add = float(adj["add"])
            out.loc[mask, "y_pred"] = out.loc[mask, "y_pred"] + add
        return out

    def _metrics_from_preds(self, df: pd.DataFrame) -> Dict[str, float]:
        mae = float(mean_absolute_error(df["y_true"], df["y_pred"]))
        rmse = _rmse_1d(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
        return {"mae": mae, "rmse": rmse}

    def _select_best(self, track: str, *, prefer_transformers: bool = True) -> Dict[str, Any]:
        items = list(self.results.get(track, {}).values())
        if not items:
            raise ValueError(f"No results for track={track}")

        def score(r: Dict[str, Any]) -> Tuple[int, float, float]:
            fam = (r.get("model_family") or "").lower()
            is_transformer = 0
            if prefer_transformers:
                is_transformer = 0 if fam == "transformer" else 1
            m = r.get("metrics", {}) or {}
            rmse = float(m.get("rmse", float("inf")))
            mae = float(m.get("mae", float("inf")))
            return (is_transformer, rmse, mae)

        return sorted(items, key=score)[0]

    def _find_baseline(self, track: str, *, variant: str = "") -> Optional[Dict[str, Any]]:
        items = list(self.results.get(track, {}).values())
        if not items:
            return None
        candidates = [r for r in items if str(r.get("model_family") or "").lower() == "baseline"]
        if not candidates:
            return None
        # Prefer same variant if provided.
        if variant:
            v = [r for r in candidates if str(r.get("variant") or "") == variant]
            if v:
                candidates = v
        # Prefer canonical baseline model name when available.
        named = [r for r in candidates if str(r.get("model_name") or "").lower() == str(self.icfg.baseline_model_name).lower()]
        if named:
            candidates = named
        # Best baseline by RMSE then MAE.
        def score(r: Dict[str, Any]) -> Tuple[float, float]:
            m = r.get("metrics", {}) or {}
            return (float(m.get("rmse", float("inf"))), float(m.get("mae", float("inf"))))

        return sorted(candidates, key=score)[0]

    def _read_preds(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Predictions parquet not found: {p}")
        df = pd.read_parquet(p)
        for c in ("timestamp", "horizon", "y_true", "y_pred"):
            if c not in df.columns:
                raise ValueError(f"Predictions parquet missing '{c}': {p}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _align(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
        key_cols = ["timestamp", "horizon"]
        if "ZONEID" in df_a.columns and "ZONEID" in df_b.columns:
            key_cols = ["ZONEID"] + key_cols
        return df_a.merge(df_b, on=key_cols, suffixes=("_a", "_b"))

    def _compute_net_load(self, load_preds: pd.DataFrame, solar_preds: pd.DataFrame) -> Dict[str, Any]:
        merged = self._align(load_preds, solar_preds)
        if len(merged) == 0:
            return {"computed": False, "reason": "No overlapping timestamps/horizons (or zones) between load and solar predictions."}

        merged = merged.copy()
        merged["net_true"] = merged["y_true_a"] - merged["y_true_b"]
        merged["net_pred"] = merged["y_pred_a"] - merged["y_pred_b"]

        mae = float(mean_absolute_error(merged["net_true"], merged["net_pred"]))
        rmse = _rmse_1d(merged["net_true"].to_numpy(), merged["net_pred"].to_numpy())

        out_path = self.run_dir / "integrated" / "net_load_predictions.parquet"
        merged.rename(
            columns={
                "y_true_a": "y_true_load",
                "y_pred_a": "y_pred_load",
                "y_true_b": "y_true_solar",
                "y_pred_b": "y_pred_solar",
            },
            inplace=True,
        )
        merged.to_parquet(out_path)
        return {"computed": True, "path": out_path.as_posix(), "n_rows": int(len(merged)), "metrics": {"mae": mae, "rmse": rmse}}

    def _plot_triple_axis_week(self, net_df: pd.DataFrame) -> Optional[str]:
        if "timestamp" not in net_df.columns:
            return None

        df = net_df.sort_values("timestamp")
        # use horizon=1 for a readable timeline
        df = df[df["horizon"] == 1].copy()
        if len(df) < 24:
            return None

        start = df["timestamp"].min().floor("D")
        end = start + pd.Timedelta(days=self.icfg.timeline_days)
        week = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
        if len(week) < 24:
            return None

        t = week["timestamp"]
        load = week["y_true_load"]
        solar = week["y_true_solar"]
        net = week["net_true"]

        fig, ax1 = plt.subplots()
        ax1.plot(t, load, label="Load (Actual)")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Load")

        ax2 = ax1.twinx()
        ax2.plot(t, solar, label="Solar (Actual)")
        ax2.set_ylabel("Solar")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.12))
        ax3.plot(t, net, label="Net-Load (Actual)")
        ax3.set_ylabel("Net-Load")

        ax1.set_title("Load, Solar, Net-Load (Test Week, Horizon=1)")
        fig.autofmt_xdate()

        lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")

        out_path = self.run_dir / "plots" / "triple_axis_week.png"
        fig.tight_layout()
        fig.savefig(out_path.as_posix(), dpi=200)
        plt.close(fig)
        return out_path.as_posix()

    def _plot_error_boxplot(self, base: pd.DataFrame, tr: pd.DataFrame, *, title: str, out_path: Path) -> str:
        merged = self._align(base, tr)
        if len(merged) == 0:
            raise ValueError("No overlap between baseline and transformer predictions for error plot.")

        ae_base = np.abs(merged["y_true_a"].to_numpy() - merged["y_pred_a"].to_numpy())
        ae_tr = np.abs(merged["y_true_b"].to_numpy() - merged["y_pred_b"].to_numpy())

        if len(ae_base) > self.icfg.max_points_boxplot:
            rs = np.random.RandomState(42)
            idx = rs.choice(len(ae_base), size=self.icfg.max_points_boxplot, replace=False)
            ae_base = ae_base[idx]
            ae_tr = ae_tr[idx]

        fig, ax = plt.subplots()
        ax.boxplot([ae_base, ae_tr], labels=["Baseline", "Transformer"], showfliers=False)
        ax.set_title(title)
        ax.set_ylabel("Absolute Error")
        fig.tight_layout()
        fig.savefig(out_path.as_posix(), dpi=200)
        plt.close(fig)
        return out_path.as_posix()

    def _plot_before_after_overlay(self, merged: pd.DataFrame, *, title: str, out_path: Path, max_points: int = 500) -> str:
        """
        merged must contain: timestamp, y_true_parent, y_pred_parent, y_pred_child
        Plots a time-ordered slice (horizon=1 recommended by caller).
        """
        df = merged.sort_values("timestamp").copy()
        if len(df) > max_points:
            df = df.iloc[:max_points].copy()

        t = pd.to_datetime(df["timestamp"])
        fig, ax = plt.subplots()
        ax.plot(t, df["y_true_parent"], label="Actual")
        ax.plot(t, df["y_pred_parent"], label="Before (parent)")
        ax.plot(t, df["y_pred_child"], label="After (child)")
        ax.set_title(title)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        fig.autofmt_xdate()
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_path.as_posix(), dpi=200)
        plt.close(fig)
        return out_path.as_posix()

    def _plot_residual_diff_hist(self, merged: pd.DataFrame, *, title: str, out_path: Path, max_points: int = 50000) -> str:
        """
        merged must contain: y_true_parent, y_pred_parent, y_pred_child
        residual_diff = |err_child| - |err_parent| (negative means improved).
        """
        err_parent = (merged["y_true_parent"] - merged["y_pred_parent"]).to_numpy()
        err_child = (merged["y_true_parent"] - merged["y_pred_child"]).to_numpy()
        diff = np.abs(err_child) - np.abs(err_parent)
        if len(diff) > max_points:
            rs = np.random.RandomState(42)
            diff = diff[rs.choice(len(diff), size=max_points, replace=False)]

        fig, ax = plt.subplots()
        ax.hist(diff, bins=60)
        ax.set_title(title)
        ax.set_xlabel("|err_after| - |err_before| (negative = improvement)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_path.as_posix(), dpi=200)
        plt.close(fig)
        return out_path.as_posix()

    def _write_before_after_report(
        self,
        *,
        track: str,
        child_run_id: str,
        parent_pred_path: str,
        child_pred_path: str,
        out_dir: Path,
    ) -> str:
        out_dir.mkdir(parents=True, exist_ok=True)

        df_p = self._read_preds(parent_pred_path)
        df_c = self._read_preds(child_pred_path)

        # Align on timestamp+horizon(+ZONEID)
        key_cols = ["timestamp", "horizon"]
        if "ZONEID" in df_p.columns and "ZONEID" in df_c.columns:
            key_cols = ["ZONEID"] + key_cols

        merged = df_p.merge(df_c, on=key_cols, suffixes=("_parent", "_child"))
        if len(merged) == 0:
            report_path = out_dir / "before_after_report.md"
            report_path.write_text(
                f"# Before vs After (Plan Edit)\n\nNo overlap between parent and child predictions for track `{track}`.\n",
                encoding="utf-8",
            )
            return report_path.as_posix()

        # Prefer horizon=1 for time plots (same style as other thesis plots)
        merged_h1 = merged[merged["horizon"] == 1].copy()
        if len(merged_h1) == 0:
            merged_h1 = merged.copy()

        # Metrics (using parent's y_true for both)
        y_true = merged["y_true_parent"].to_numpy()
        y_pred_parent = merged["y_pred_parent"].to_numpy()
        y_pred_child = merged["y_pred_child"].to_numpy()

        mae_parent = float(mean_absolute_error(y_true, y_pred_parent))
        rmse_parent = _rmse_1d(y_true, y_pred_parent)
        mae_child = float(mean_absolute_error(y_true, y_pred_child))
        rmse_child = _rmse_1d(y_true, y_pred_child)

        # Plots
        overlay_path = self._plot_before_after_overlay(
            merged_h1,
            title=f"{track.upper()} Before vs After (H=1) | child={child_run_id}",
            out_path=out_dir / "before_after_overlay.png",
        )
        resid_path = self._plot_residual_diff_hist(
            merged,
            title=f"{track.upper()} Residual Diff | child={child_run_id}",
            out_path=out_dir / "residual_diff_hist.png",
        )

        # Save aligned comparison parquet (useful for thesis tables)
        cmp_path = out_dir / "before_after_aligned.parquet"
        merged.to_parquet(cmp_path)

        report_path = out_dir / "before_after_report.md"
        report_path.write_text(
            "\n".join(
                [
                    f"# Before vs After (Plan Edit) | {track.upper()}",
                    "",
                    f"- parent_run_id: `{self.run_id}`",
                    f"- child_run_id: `{child_run_id}`",
                    "",
                    "## Metrics (same aligned set)",
                    f"- Before: MAE={mae_parent:.6f} RMSE={rmse_parent:.6f}",
                    f"- After:  MAE={mae_child:.6f} RMSE={rmse_child:.6f}",
                    f"- Delta:  MAE={mae_child - mae_parent:+.6f} RMSE={rmse_child - rmse_parent:+.6f}",
                    "",
                    "## Artifacts",
                    f"- Overlay plot: `{overlay_path}`",
                    f"- Residual diff hist: `{resid_path}`",
                    f"- Aligned parquet: `{cmp_path.as_posix()}`",
                    "",
                    "Interpretation tip: negative values in the residual-diff histogram mean the child run improved absolute error compared to the parent.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return report_path.as_posix()

    def _export_audit(self) -> str:
        p = Path(self.icfg.message_pool_path)
        if not p.exists():
            return ""

        prep_events = []
        train_cmds = []

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("run_id") != self.run_id:
                    continue
                topic = obj.get("topic", "")
                sender = obj.get("sender", "")
                kind = obj.get("kind", "")
                payload = obj.get("payload", {}) or {}

                if sender.lower().startswith("preparation") and payload.get("event") in ("ready", "not_ready"):
                    prep_events.append(obj)
                if topic.startswith(f"train/{self.run_id}") and kind == "command":
                    train_cmds.append(obj)

        out_path = self.run_dir / "reports" / "system_performance_audit.md"
        lines = []
        lines.append(f"# System Performance Audit ({self.run_id})\n\n")

        lines.append("## Preparation trace\n")
        if not prep_events:
            lines.append("- No preparation events found.\n")
        else:
            for e in prep_events:
                payload = e.get("payload", {}) or {}
                lines.append(f"- `{e.get('created_at')}` track=`{payload.get('track')}` event=`{payload.get('event')}`\n")
                if payload.get("event") == "ready":
                    paths = payload.get("paths", {}) or {}
                    lines.append(f"  - train_scaled: `{paths.get('train_scaled','')}`\n")
                    lines.append(f"  - test_scaled: `{paths.get('test_scaled','')}`\n")
                if payload.get("event") == "not_ready":
                    lines.append(f"  - error: `{payload.get('error')}`\n")

        lines.append("\n## Training plan trace\n")
        if not train_cmds:
            lines.append("- No train commands found.\n")
        else:
            for t in train_cmds:
                payload = t.get("payload", {}) or {}
                model = payload.get("model_name") or payload.get("model_type")
                lines.append(f"- `{t.get('created_at')}` track=`{payload.get('track')}` model=`{model}`\n")
                hp = payload.get("hyperparams") or {}
                if hp:
                    lines.append(f"  - hyperparams: `{hp}`\n")

        out_path.write_text("".join(lines), encoding="utf-8")
        return out_path.as_posix()

    def _export_interaction_audit(self, *, child_info: Dict[str, Any]) -> str:
        """
        Phase D2 thesis-style audit:
        - gates requested/approved
        - raw user_feedback
        - parsed directives
        - evaluation results + deltas when possible
        """
        log_path = Path(self.icfg.message_pool_path)
        if not log_path.exists():
            return ""

        rows = []
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("run_id") != self.run_id:
                    continue
                rows.append(obj)

        def by_topic(prefix: str):
            return [r for r in rows if str(r.get("topic", "")).startswith(prefix)]

        gates = [r for r in by_topic(f"gate/{self.run_id}") if (r.get("payload") or {}).get("event") in ("gate_requested", "gate_approved", "gate_rejected")]
        feedback = by_topic(f"user_feedback/{self.run_id}")
        directives = by_topic(f"directive/{self.run_id}")
        sys_directives = by_topic(f"system_directive/{self.run_id}")
        acks = by_topic(f"directive_ack/{self.run_id}")
        rejected = by_topic(f"directive_rejected/{self.run_id}")
        gate_decisions = by_topic(f"human_gate_decision/{self.run_id}")
        evals = [r for r in by_topic(f"evaluate/{self.run_id}") if (r.get("payload") or {}).get("event") == "evaluation_complete"]

        out_path = self.run_dir / "reports" / "interaction_audit.md"
        lines = []
        lines.append(f"# Interaction Audit ({self.run_id})\n\n")

        lines.append("## Gates\n")
        if not gates:
            lines.append("- No gates recorded.\n")
        else:
            for g in gates:
                p = g.get("payload") or {}
                lines.append(f"- seq={g.get('seq')} stage=`{p.get('stage')}` gate_id=`{p.get('gate_id')}` event=`{p.get('event')}`\n")

        lines.append("\n## Human Inputs (Raw)\n")
        if not feedback:
            lines.append("- No user_feedback messages found.\n")
        else:
            for r in feedback:
                p = r.get("payload") or {}
                lines.append(f"- gate_id=`{p.get('gate_id')}` stage=`{p.get('stage')}` text=`{p.get('human_text')}`\n")

        lines.append("\n## Parsed Directives\n")
        if not directives:
            lines.append("- No directives found.\n")
        else:
            for d in directives:
                p = d.get("payload") or {}
                lines.append(f"- gate_id=`{p.get('gate_id')}` stage=`{p.get('stage')}` structured=`{p.get('structured_command')}`\n")

        lines.append("\n## System Directives (Phase D3)\n")
        if not sys_directives:
            lines.append("- No system_directive messages found.\n")
        else:
            for d in sys_directives:
                p = d.get("payload") or {}
                lines.append(
                    f"- stage=`{p.get('stage')}` gate_id=`{p.get('gate_id')}` overrides=`{p.get('overrides')}` reason=`{p.get('reason','')}`\n"
                )

        lines.append("\n## Directive Acks (Phase D3)\n")
        if not acks:
            lines.append("- No directive_ack messages found.\n")
        else:
            for a in acks:
                p = a.get("payload") or {}
                lines.append(f"- stage=`{p.get('stage')}` directive_id=`{p.get('directive_id')}` applied=`{p.get('applied_overrides')}`\n")

        lines.append("\n## Directive Rejections (Phase D4)\n")
        if not rejected:
            lines.append("- No directive_rejected messages found.\n")
        else:
            for r in rejected:
                p = r.get("payload") or {}
                lines.append(f"- stage=`{p.get('stage')}` directive_id=`{p.get('directive_id')}` rejected=`{p.get('rejected_overrides')}`\n")

        lines.append("\n## Human Gate Decisions (Phase D3)\n")
        if not gate_decisions:
            lines.append("- No human_gate_decision messages found.\n")
        else:
            for g in gate_decisions:
                p = g.get("payload") or {}
                lines.append(f"- stage=`{p.get('stage')}` gate_id=`{p.get('gate_id')}` decision=`{p.get('decision')}` text=`{p.get('raw_text')}`\n")

        lines.append("\n## Evaluation Results (This Run)\n")
        if not evals:
            lines.append("- No evaluation results found.\n")
        else:
            for e in evals:
                p = e.get("payload") or {}
                m = p.get("metrics") or {}
                lines.append(
                    f"- track=`{p.get('track')}` model=`{p.get('model_name')}` family=`{p.get('model_family','')}` "
                    f"variant=`{p.get('variant','')}` cmp=`{p.get('comparison_group_id','')}` child_run_id=`{p.get('child_run_id','')}` "
                    f"MAE={m.get('mae')} RMSE={m.get('rmse')}\n"
                )

        # Best-effort comparative deltas if both variants are present for a track.
        try:
            by_track_var: Dict[str, Dict[str, Dict[str, float]]] = {"load": {}, "solar": {}}
            for e in evals:
                p = e.get("payload") or {}
                tr = p.get("track")
                if tr not in ("load", "solar"):
                    continue
                var = p.get("variant") or ""
                if var not in ("proposed", "human_adjusted"):
                    continue
                m = p.get("metrics") or {}
                if "mae" in m and "rmse" in m:
                    by_track_var[tr][var] = {"mae": float(m["mae"]), "rmse": float(m["rmse"])}
            lines.append("\n## Comparative Deltas (If Both Branches Exist)\n")
            any_delta = False
            for tr in ("load", "solar"):
                if "proposed" in by_track_var[tr] and "human_adjusted" in by_track_var[tr]:
                    any_delta = True
                    p0 = by_track_var[tr]["proposed"]
                    p1 = by_track_var[tr]["human_adjusted"]
                    mae_imp = ((p0["mae"] - p1["mae"]) / p0["mae"] * 100.0) if p0["mae"] else 0.0
                    rmse_imp = ((p0["rmse"] - p1["rmse"]) / p0["rmse"] * 100.0) if p0["rmse"] else 0.0
                    lines.append(f"- {tr}: MAE {p0['mae']} -> {p1['mae']} ({mae_imp:.2f}%), RMSE {p0['rmse']} -> {p1['rmse']} ({rmse_imp:.2f}%)\n")
            if not any_delta:
                lines.append("- No proposed vs human_adjusted pair found in this run.\n")
        except Exception:
            pass

        # Link known comparison artifacts if present.
        plan_diff = self.run_dir / "reports" / "plan_edit_diff.md"
        if plan_diff.exists():
            lines.append("\n## Plan Edit Diff\n")
            lines.append(f"- `{plan_diff.as_posix()}`\n")

        if child_info and child_info.get("child_run_id"):
            lines.append("\n## Post-Process Adjustment\n")
            lines.append(f"- child_run_id: `{child_info.get('child_run_id')}`\n")
            lines.append(f"- target: `{child_info.get('target')}`\n")
            lines.append(f"- adjustment: `{child_info.get('adjustment')}`\n")
            lines.append(f"- comparison: `logs/runs/{self.run_id}/child_runs/{child_info.get('child_run_id')}/reports/postprocess_comparison.md`\n")

        out_path.write_text("".join(lines), encoding="utf-8")
        return out_path.as_posix()

    def _write_reports(self, *, best_load: Dict[str, Any], best_solar: Dict[str, Any], net_info: Dict[str, Any], plots: Dict[str, str]) -> Tuple[str, str]:
        joint_path = self.run_dir / "reports" / "joint_decision_report.md"
        final_path = self.run_dir / "reports" / "final_research_summary.md"

        def fmt_model(r: Dict[str, Any]) -> str:
            fam = r.get("model_family", "")
            name = r.get("model_name") or r.get("model_type")
            return f"{name} ({fam})"

        joint_lines = [f"# Joint Load + Solar Forecast Report ({self.run_id})\n\n"]
        joint_lines.append("## Load (best)\n")
        joint_lines.append(f"- Model: `{fmt_model(best_load)}`\n")
        if best_load.get("comparison_group_id") or best_load.get("variant"):
            joint_lines.append(f"- Variant: `{best_load.get('variant','')}` cmp=`{best_load.get('comparison_group_id','')}`\n")
        joint_lines.append(f"- MAE: **{best_load['metrics'].get('mae', float('nan')):.6f}**, RMSE: **{best_load['metrics'].get('rmse', float('nan')):.6f}**\n")
        joint_lines.append(f"- Plot: `{best_load['artifacts'].get('plot_path','')}`\n")
        joint_lines.append(f"- Predictions: `{best_load['artifacts'].get('predictions_path','')}`\n")

        joint_lines.append("\n## Solar (best)\n")
        joint_lines.append(f"- Model: `{fmt_model(best_solar)}`\n")
        if best_solar.get("comparison_group_id") or best_solar.get("variant"):
            joint_lines.append(f"- Variant: `{best_solar.get('variant','')}` cmp=`{best_solar.get('comparison_group_id','')}`\n")
        joint_lines.append(f"- MAE: **{best_solar['metrics'].get('mae', float('nan')):.6f}**, RMSE: **{best_solar['metrics'].get('rmse', float('nan')):.6f}**\n")
        joint_lines.append(f"- Plot: `{best_solar['artifacts'].get('plot_path','')}`\n")
        joint_lines.append(f"- Predictions: `{best_solar['artifacts'].get('predictions_path','')}`\n")

        joint_lines.append("\n## Net-load\n")
        if not net_info.get("computed", False):
            joint_lines.append(f"- Net-load not computed: {net_info.get('reason')}\n")
        else:
            m = net_info["metrics"]
            joint_lines.append(f"- Net-load computed rows={net_info.get('n_rows')}\n")
            joint_lines.append(f"- MAE: **{m['mae']:.6f}**, RMSE: **{m['rmse']:.6f}**\n")
            joint_lines.append(f"- Artifact: `{net_info.get('path')}`\n")
            if "triple_axis_week" in plots:
                joint_lines.append(f"- Plot: `{plots['triple_axis_week']}`\n")

        joint_path.write_text("".join(joint_lines), encoding="utf-8")

        final_lines = [f"# Final Research Summary ({self.run_id})\n\n"]
        final_lines.append("## Best models (prefer transformers)\n")
        final_lines.append(f"- Load: `{fmt_model(best_load)}` MAE={best_load['metrics'].get('mae')} RMSE={best_load['metrics'].get('rmse')}\n")
        final_lines.append(f"- Solar: `{fmt_model(best_solar)}` MAE={best_solar['metrics'].get('mae')} RMSE={best_solar['metrics'].get('rmse')}\n")

        final_lines.append("\n## Net-load synthesis\n")
        if not net_info.get("computed", False):
            final_lines.append(f"- Net-load not computed: {net_info.get('reason')}\n")
        else:
            m = net_info["metrics"]
            final_lines.append(f"- Net-load MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}\n")
            final_lines.append(f"- Net-load artifact: `{net_info.get('path')}`\n")
            if "triple_axis_week" in plots:
                final_lines.append(f"- Triple-axis plot: `{plots['triple_axis_week']}`\n")

        final_lines.append("\n## Error distribution (baseline vs transformer)\n")
        final_lines.append(f"- Load plot: `{plots.get('load_error_boxplot','(not available)')}`\n")
        final_lines.append(f"- Solar plot: `{plots.get('solar_error_boxplot','(not available)')}`\n")

        final_lines.append("\n## Audit\n")
        final_lines.append(f"- System audit: `{plots.get('audit_path','')}`\n")
        if plots.get("interaction_audit"):
            final_lines.append(f"- Interaction audit: `{plots.get('interaction_audit','')}`\n")
        if plots.get("comparative_branch_report"):
            final_lines.append(f"- Comparative branch report: `{plots.get('comparative_branch_report','')}`\n")

        final_path.write_text("".join(final_lines), encoding="utf-8")
        return joint_path.as_posix(), final_path.as_posix()

    def _export_comparative_branch_report(self) -> str:
        """
        Phase D3: If both variants exist (proposed + human_adjusted), write a compact comparison report.
        """
        items_by_track = {t: list((self.results.get(t) or {}).values()) for t in ("load", "solar")}
        # Determine if we have both variants for at least one track.
        def variants_for(track: str) -> set[str]:
            return {str(r.get("variant") or "") for r in items_by_track[track] if str(r.get("variant") or "")}

        if not ({"proposed", "human_adjusted"} <= variants_for("load") or {"proposed", "human_adjusted"} <= variants_for("solar")):
            return ""

        def best_for(track: str, variant: str) -> Optional[Dict[str, Any]]:
            cand = [r for r in items_by_track[track] if str(r.get("variant") or "") == variant]
            if not cand:
                return None
            # Prefer transformers, but allow baseline fallback.
            def score(r: Dict[str, Any]) -> Tuple[int, float, float]:
                fam = str(r.get("model_family") or "").lower()
                is_transformer = 0 if fam == "transformer" else 1
                m = r.get("metrics", {}) or {}
                return (is_transformer, float(m.get("rmse", float("inf"))), float(m.get("mae", float("inf"))))

            return sorted(cand, key=score)[0]

        out_path = self.run_dir / "reports" / "comparative_branch_report.md"
        lines = [f"# Comparative Branch Report ({self.run_id})\n\n"]
        # Try to keep cmp id visible
        cmp_ids = {str(r.get("comparison_group_id") or "") for t in ("load", "solar") for r in items_by_track[t] if str(r.get("comparison_group_id") or "")}
        if cmp_ids:
            lines.append(f"- comparison_group_id(s): `{sorted(cmp_ids)}`\n\n")

        for track in ("load", "solar"):
            p0 = best_for(track, "proposed")
            p1 = best_for(track, "human_adjusted")
            lines.append(f"## {track.title()}\n")
            if not p0 or not p1:
                lines.append("- Missing one of the branches for this track.\n\n")
                continue
            m0 = p0.get("metrics", {}) or {}
            m1 = p1.get("metrics", {}) or {}
            mae0 = float(m0.get("mae", float("nan")))
            mae1 = float(m1.get("mae", float("nan")))
            rmse0 = float(m0.get("rmse", float("nan")))
            rmse1 = float(m1.get("rmse", float("nan")))
            mae_imp = ((mae0 - mae1) / mae0 * 100.0) if mae0 and mae0 == mae0 else float("nan")
            rmse_imp = ((rmse0 - rmse1) / rmse0 * 100.0) if rmse0 and rmse0 == rmse0 else float("nan")
            lines.append(f"- Proposed: model=`{p0.get('model_name')}` family=`{p0.get('model_family')}` MAE={m0.get('mae')} RMSE={m0.get('rmse')}\n")
            lines.append(f"- Human-adjusted: model=`{p1.get('model_name')}` family=`{p1.get('model_family')}` MAE={m1.get('mae')} RMSE={m1.get('rmse')}\n")
            lines.append(f"- Improvement: MAE={mae_imp:.2f}% RMSE={rmse_imp:.2f}%\n\n")

        # Net-load comparison if both branches have overlapping timestamps.
        try:
            lp0 = best_for("load", "proposed")
            sp0 = best_for("solar", "proposed")
            lp1 = best_for("load", "human_adjusted")
            sp1 = best_for("solar", "human_adjusted")
            if lp0 and sp0 and lp1 and sp1:
                lines.append("## Net-load (if overlap exists)\n")
                info0 = {"computed": False, "reason": "missing"}
                info1 = {"computed": False, "reason": "missing"}
                if lp0.get("artifacts", {}).get("predictions_path") and sp0.get("artifacts", {}).get("predictions_path"):
                    dfL = self._read_preds(lp0["artifacts"]["predictions_path"])
                    dfS = self._read_preds(sp0["artifacts"]["predictions_path"])
                    info0 = self._compute_net_load(dfL, dfS)
                if lp1.get("artifacts", {}).get("predictions_path") and sp1.get("artifacts", {}).get("predictions_path"):
                    dfL = self._read_preds(lp1["artifacts"]["predictions_path"])
                    dfS = self._read_preds(sp1["artifacts"]["predictions_path"])
                    info1 = self._compute_net_load(dfL, dfS)
                if info0.get("computed") and info1.get("computed"):
                    m0 = info0.get("metrics", {}) or {}
                    m1 = info1.get("metrics", {}) or {}
                    mae0 = float(m0.get("mae", float("nan")))
                    mae1 = float(m1.get("mae", float("nan")))
                    rmse0 = float(m0.get("rmse", float("nan")))
                    rmse1 = float(m1.get("rmse", float("nan")))
                    mae_imp = ((mae0 - mae1) / mae0 * 100.0) if mae0 and mae0 == mae0 else float("nan")
                    rmse_imp = ((rmse0 - rmse1) / rmse0 * 100.0) if rmse0 and rmse0 == rmse0 else float("nan")
                    lines.append(f"- Proposed: MAE={m0.get('mae')} RMSE={m0.get('rmse')}\n")
                    lines.append(f"- Human-adjusted: MAE={m1.get('mae')} RMSE={m1.get('rmse')}\n")
                    lines.append(f"- Improvement: MAE={mae_imp:.2f}% RMSE={rmse_imp:.2f}%\n\n")
                else:
                    lines.append(f"- Not computed for both branches: proposed_reason=`{info0.get('reason')}`, adjusted_reason=`{info1.get('reason')}`\n\n")
        except Exception:
            pass

        out_path.write_text("".join(lines), encoding="utf-8")
        return out_path.as_posix()

    def _ready_to_finalize(self) -> bool:
        # Minimum: at least one transformer result per track.
        def has_transformer(track: str) -> bool:
            for r in (self.results.get(track, {}) or {}).values():
                fam = str(r.get("model_family") or "").lower()
                name = str(r.get("model_name") or "").lower()
                if fam == "transformer" and name in {n.lower() for n in self.icfg.transformer_model_names}:
                    return True
            return False

        return has_transformer("load") and has_transformer("solar")

    def tick(self) -> None:
        for msg in self.poll(max_messages=200):
            if msg.topic.startswith(f"system/{self.run_id}") and msg.kind == "event":
                if (msg.payload or {}).get("mode") == "research_interactive":
                    self._interactive = True

            if msg.kind != "result":
                # Phase D2: only consume structured directives (no raw-English parsing here).
                if msg.topic.startswith(f"directive/{self.run_id}") and msg.kind == "event":
                    payload = msg.payload or {}
                    if payload.get("gate_id") == self.POSTPROCESS_GATE_ID and payload.get("event") == "directive":
                        cmd = payload.get("structured_command") or {}
                        if isinstance(cmd, dict):
                            self._post_action = dict(cmd)
                            self._post_done = True
                continue

            if msg.payload.get("event") != "evaluation_complete":
                continue

            track = msg.payload.get("track")
            if track not in ("load", "solar"):
                continue

            model_name = msg.payload.get("model_name") or msg.payload.get("model_type") or "unknown"
            fam = msg.payload.get("model_family") or ("transformer" if msg.payload.get("strategy") else "baseline")
            variant = msg.payload.get("variant", "") or ""
            child_run_id = msg.payload.get("child_run_id", "") or ""
            # Phase D3: avoid overwriting when we run proposed + human_adjusted using the same model name.
            # Key is internal only; reports use the fields.
            key = f"{model_name}|{fam}|{variant}|{child_run_id or 'root'}"
            self.results[track][key] = {
                "track": track,
                "model_family": fam,
                "model_name": model_name,
                "strategy": msg.payload.get("strategy", ""),
                "metrics": msg.payload.get("metrics", {}) or {},
                "artifacts": msg.payload.get("artifacts", {}) or {},
                "device": msg.payload.get("device", ""),
                "child_run_id": child_run_id,
                "parent_run_id": msg.payload.get("parent_run_id", "") or "",
                "comparison_group_id": msg.payload.get("comparison_group_id", "") or "",
                "variant": variant,
            }

        if self._finalized or not self._ready_to_finalize():
            return

        # Detect interactive mode lazily (in case system/prepare msgs were emitted before we subscribed).
        if not self._interactive:
            self._interactive = self._detect_interactive()

        best_load = self._select_best("load", prefer_transformers=True)
        best_solar = self._select_best("solar", prefer_transformers=True)

        net_info: Dict[str, Any] = {"computed": False, "reason": "Missing prediction artifacts."}
        plots: Dict[str, str] = {}

        # Net-load (best load + best solar)
        lp = best_load["artifacts"].get("predictions_path")
        sp = best_solar["artifacts"].get("predictions_path")
        if lp and sp:
            try:
                dfL = self._read_preds(lp)
                dfS = self._read_preds(sp)
                net_info = self._compute_net_load(dfL, dfS)
                if net_info.get("computed"):
                    net_df = pd.read_parquet(net_info["path"])
                    triple = self._plot_triple_axis_week(net_df)
                    if triple:
                        plots["triple_axis_week"] = triple
            except Exception as e:
                net_info = {"computed": False, "reason": str(e)}

        # Error distribution plots (baseline vs best transformer) per track, if baseline exists.
        for track in ("load", "solar"):
            tr_best = self._select_best(track, prefer_transformers=True)
            base = self._find_baseline(track, variant=str(tr_best.get("variant") or ""))
            if not base:
                continue
            bp = base["artifacts"].get("predictions_path")
            tp = tr_best["artifacts"].get("predictions_path")
            if not bp or not tp:
                continue
            try:
                dfb = self._read_preds(bp)
                dft = self._read_preds(tp)
                out = self.run_dir / "plots" / f"{track}_error_distribution_boxplot.png"
                pth = self._plot_error_boxplot(dfb, dft, title=f"{track.upper()} | Abs Error: Baseline vs Transformer", out_path=out)
                plots[f"{track}_error_boxplot"] = pth
            except Exception as e:
                plots[f"{track}_error_boxplot"] = f"(error: {e})"

        audit_path = self._export_audit()
        if audit_path:
            plots["audit_path"] = audit_path

        comp_path = self._export_comparative_branch_report()
        if comp_path:
            plots["comparative_branch_report"] = comp_path

        # If this run produced any child-run artifacts (plan edit), generate before/after comparisons.
        # We compare parent vs child predictions for the same track/model_name when available.
        try:
            for track in ("load", "solar"):
                for _name, r in (self.results.get(track) or {}).items():
                    child_run_id = (r.get("child_run_id") or "").strip()
                    parent_run_id = (r.get("parent_run_id") or "").strip()
                    if not child_run_id or not parent_run_id:
                        continue
                    # Only compare within this parent run instance.
                    if parent_run_id != self.run_id:
                        continue
                    child_pred = (r.get("artifacts") or {}).get("predictions_path", "")
                    if not child_pred:
                        continue
                    # Find a parent result for the same track/model_name with no child id.
                    parent_pred = ""
                    for _pn, pr in (self.results.get(track) or {}).items():
                        if (pr.get("child_run_id") or ""):
                            continue
                        if pr.get("model_name") != r.get("model_name"):
                            continue
                        parent_pred = (pr.get("artifacts") or {}).get("predictions_path", "")
                        if parent_pred:
                            break
                    cmp_dir = self.run_dir / "child_runs" / child_run_id / "reports"
                    if not parent_pred:
                        # If the user edited the plan before the first training run, we have no "before" forecasts.
                        # Still write an explicit report so the thesis appendix is honest and complete.
                        cmp_dir.mkdir(parents=True, exist_ok=True)
                        rep_path = cmp_dir / "before_after_report.md"
                        rep_path.write_text(
                            "\n".join(
                                [
                                    f"# Before vs After (Plan Edit) | {track.upper()}",
                                    "",
                                    f"- parent_run_id: `{self.run_id}`",
                                    f"- child_run_id: `{child_run_id}`",
                                    "",
                                    "No parent forecast artifacts were produced for the same model within this run,",
                                    "so a forecast-level before/after comparison cannot be generated.",
                                    "",
                                    "This typically happens when the plan is edited before the first training execution.",
                                    "You can get full before/after forecast diffs by: approve once to run, then change plan and re-run.",
                                    "",
                                    f"- child_predictions: `{child_pred}`",
                                ]
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        plots[f"{track}_before_after_report"] = rep_path.as_posix()
                    else:
                        rep = self._write_before_after_report(
                            track=track,
                            child_run_id=child_run_id,
                            parent_pred_path=parent_pred,
                            child_pred_path=child_pred,
                            out_dir=cmp_dir,
                        )
                        plots[f"{track}_before_after_report"] = rep
        except Exception:
            pass

        # If interactive, request postprocess gate once, and wait for feedback before finalizing.
        if self._interactive and not self._post_done:
            if not self._post_gate_requested:
                self._post_gate_requested = True
                self._cached_best = {"load": best_load, "solar": best_solar}
                self._cached_net_info = dict(net_info)
                self._cached_plots = dict(plots)
                self.publish(
                    topic_type="gate",
                    kind="event",
                    payload={
                        "event": "gate_requested",
                        "stage": "postprocess",
                        "gate_id": self.POSTPROCESS_GATE_ID,
                        "prompt": self._postprocess_prompt(best_load=best_load, best_solar=best_solar, net_info=net_info),
                    },
                )
            return

        # Apply postprocess adjustment if requested (writes child-run artifacts, does not overwrite originals).
        child_info: Dict[str, Any] = {}
        if self._interactive and self._post_done:
            action = self._post_action or {"action": "accept"}
            if action.get("action") not in ("accept", "reject", "scale", "bias"):
                action = {"action": "accept"}

            if action.get("action") in ("scale", "bias"):
                child_id = f"child_{self._utc_compact()}_postprocess"
                child_dir = self.run_dir / "child_runs" / child_id
                (child_dir / "predictions").mkdir(parents=True, exist_ok=True)
                (child_dir / "reports").mkdir(parents=True, exist_ok=True)

                target = str(action.get("target"))
                best = best_load if target == "load" else best_solar
                pred_path = best["artifacts"].get("predictions_path")
                if pred_path:
                    df = self._read_preds(pred_path)
                    df_adj = self._apply_adjustment(df, action)
                    df_adj["run_id"] = child_id
                    df_adj["parent_run_id"] = self.run_id
                    df_adj["variant"] = "adjusted"
                    df_adj["adjustment"] = json.dumps(action, ensure_ascii=False)

                    out_pred = child_dir / "predictions" / f"{target}_adjusted_predictions.parquet"
                    df_adj.to_parquet(out_pred)

                    adj_metrics = self._metrics_from_preds(df_adj)
                    child_info = {
                        "child_run_id": child_id,
                        "parent_run_id": self.run_id,
                        "target": target,
                        "adjustment": action,
                        "adjusted_predictions_path": out_pred.as_posix(),
                        "adjusted_metrics": adj_metrics,
                    }

                    # Write a concise comparison report
                    rep = child_dir / "reports" / "postprocess_comparison.md"
                    before = best.get("metrics", {}) or {}
                    rep.write_text(
                        "\n".join(
                            [
                                f"# Post-Process Comparison ({child_id})",
                                "",
                                f"- parent_run_id: `{self.run_id}`",
                                f"- target: `{target}`",
                                f"- adjustment: `{action}`",
                                "",
                                "## Metrics (before vs after)",
                                f"- before MAE={before.get('mae')} RMSE={before.get('rmse')}",
                                f"- after  MAE={adj_metrics.get('mae')} RMSE={adj_metrics.get('rmse')}",
                                "",
                                f"- adjusted_predictions: `{out_pred.as_posix()}`",
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                else:
                    child_info = {"computed": False, "reason": "missing predictions_path for selected target", "adjustment": action}

            # reject == accept (no changes), but we still proceed to finalize

        # Now produce final reports (optionally referencing child adjustment in the final summary via net_load payload).
        joint_report, final_report = self._write_reports(best_load=best_load, best_solar=best_solar, net_info=net_info, plots=plots)
        self._finalized = True

        interaction_audit_path = self._export_interaction_audit(child_info=child_info)
        if interaction_audit_path:
            # Keep it discoverable from existing audit report too.
            plots["interaction_audit"] = interaction_audit_path

        # Backward compatible event
        self.publish(
            topic_type="integrate",
            kind="result",
            payload={"event": "joint_report_ready", "report_path": joint_report, "net_load": net_info, "postprocess": child_info, "interaction_audit": interaction_audit_path},
        )
        # Phase C3 event
        self.publish(
            topic_type="integrate",
            kind="result",
            payload={"event": "final_research_summary_ready", "report_path": final_report, "net_load": net_info, "postprocess": child_info, "interaction_audit": interaction_audit_path},
        )

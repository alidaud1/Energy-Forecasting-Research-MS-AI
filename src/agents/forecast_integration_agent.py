from __future__ import annotations

import json
from dataclasses import dataclass
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

    def __init__(self, cfg, pool, icfg: Optional[IntegrationConfig] = None):
        super().__init__(cfg, pool)
        self.subscribe(topics=["evaluate"], start_from="latest")

        self.icfg = icfg or IntegrationConfig()

        self.run_dir = Path("logs") / "runs" / self.run_id
        (self.run_dir / "integrated").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)

        # results[track][model_name] = payload summary
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {"load": {}, "solar": {}}
        self._finalized = False

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
        joint_lines.append(f"- MAE: **{best_load['metrics'].get('mae', float('nan')):.6f}**, RMSE: **{best_load['metrics'].get('rmse', float('nan')):.6f}**\n")
        joint_lines.append(f"- Plot: `{best_load['artifacts'].get('plot_path','')}`\n")
        joint_lines.append(f"- Predictions: `{best_load['artifacts'].get('predictions_path','')}`\n")

        joint_lines.append("\n## Solar (best)\n")
        joint_lines.append(f"- Model: `{fmt_model(best_solar)}`\n")
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

        final_path.write_text("".join(final_lines), encoding="utf-8")
        return joint_path.as_posix(), final_path.as_posix()

    def _ready_to_finalize(self) -> bool:
        # Minimum: at least one transformer result per track.
        def has_transformer(track: str) -> bool:
            return any(m in self.results[track] for m in self.icfg.transformer_model_names)

        return has_transformer("load") and has_transformer("solar")

    def tick(self) -> None:
        for msg in self.poll(max_messages=200):
            if msg.kind != "result":
                continue
            if msg.payload.get("event") != "evaluation_complete":
                continue

            track = msg.payload.get("track")
            if track not in ("load", "solar"):
                continue

            model_name = msg.payload.get("model_name") or msg.payload.get("model_type") or "unknown"
            self.results[track][model_name] = {
                "track": track,
                "model_family": msg.payload.get("model_family") or ("transformer" if msg.payload.get("strategy") else "baseline"),
                "model_name": model_name,
                "strategy": msg.payload.get("strategy", ""),
                "metrics": msg.payload.get("metrics", {}) or {},
                "artifacts": msg.payload.get("artifacts", {}) or {},
                "device": msg.payload.get("device", ""),
            }

        if self._finalized or not self._ready_to_finalize():
            return

        self._finalized = True

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
            base = self.results[track].get(self.icfg.baseline_model_name)
            tr_best = self._select_best(track, prefer_transformers=True)
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

        joint_report, final_report = self._write_reports(best_load=best_load, best_solar=best_solar, net_info=net_info, plots=plots)

        # Backward compatible event
        self.publish(
            topic_type="integrate",
            kind="result",
            payload={"event": "joint_report_ready", "report_path": joint_report, "net_load": net_info},
        )
        # Phase C3 event
        self.publish(
            topic_type="integrate",
            kind="result",
            payload={"event": "final_research_summary_ready", "report_path": final_report, "net_load": net_info},
        )


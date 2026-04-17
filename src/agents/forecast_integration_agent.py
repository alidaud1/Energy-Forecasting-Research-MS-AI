from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .base_agent import BaseAgent


class ForecastIntegrationAgent(BaseAgent):
    """
    Subscribes to evaluation results from both tracks.
    Produces:
    - joint decision report (Markdown)
    - optional net-load predictions parquet if timestamps overlap
    """

    def __init__(self, cfg, pool):
        super().__init__(cfg, pool)
        self.subscribe(topics=["evaluate"], start_from="latest")

        self.run_dir = Path("logs") / "runs" / self.run_id
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "integrated").mkdir(parents=True, exist_ok=True)

        self._results: Dict[str, Dict[str, Any]] = {}

    def _try_compute_net_load(self) -> Optional[dict]:
        if "load" not in self._results or "solar" not in self._results:
            return None

        load_pred_path = self._results["load"]["artifacts"].get("predictions_path", "")
        solar_pred_path = self._results["solar"]["artifacts"].get("predictions_path", "")
        if not load_pred_path or not solar_pred_path:
            return {"computed": False, "reason": "Missing prediction artifacts for one or both tracks."}

        dfL = pd.read_parquet(load_pred_path)
        dfS = pd.read_parquet(solar_pred_path)

        key_cols = ["timestamp", "horizon"]
        if "ZONEID" in dfL.columns and "ZONEID" in dfS.columns:
            key_cols = ["ZONEID"] + key_cols

        merged = dfL.merge(dfS, on=key_cols, suffixes=("_load", "_solar"))
        if len(merged) == 0:
            return {"computed": False, "reason": "No overlapping timestamps/horizons between load and solar predictions."}

        merged["net_true"] = merged["y_true_load"] - merged["y_true_solar"]
        merged["net_pred"] = merged["y_pred_load"] - merged["y_pred_solar"]

        out_path = self.run_dir / "integrated" / "net_load_predictions.parquet"
        merged.to_parquet(out_path)
        return {"computed": True, "path": out_path.as_posix(), "n_rows": int(len(merged))}

    def _write_report(self, net_info: Optional[dict]) -> str:
        report_path = self.run_dir / "reports" / "joint_decision_report.md"

        load = self._results.get("load")
        solar = self._results.get("solar")

        lines = [f"# Joint Load + Solar Forecast Report ({self.run_id})\n\n"]

        lines.append("## Load forecasting result\n")
        if load:
            lines.append(f"- Model: `{load.get('model_name')}` | Strategy: `{load.get('strategy')}`\n")
            lines.append(f"- MAE: **{load['metrics']['mae']:.6f}**, RMSE: **{load['metrics']['rmse']:.6f}**\n")
            lines.append(f"- Plot: `{load['artifacts'].get('plot_path','')}`\n")
            lines.append(f"- Predictions: `{load['artifacts'].get('predictions_path','')}`\n")
        else:
            lines.append("- Not available.\n")

        lines.append("\n## Solar forecasting result\n")
        if solar:
            lines.append(f"- Model: `{solar.get('model_name')}` | Strategy: `{solar.get('strategy')}`\n")
            lines.append(f"- MAE: **{solar['metrics']['mae']:.6f}**, RMSE: **{solar['metrics']['rmse']:.6f}**\n")
            lines.append(f"- Plot: `{solar['artifacts'].get('plot_path','')}`\n")
            lines.append(f"- Predictions: `{solar['artifacts'].get('predictions_path','')}`\n")
        else:
            lines.append("- Not available.\n")

        lines.append("\n## Net-load integration\n")
        if not net_info:
            lines.append("- Net-load not computed (missing one of the tracks).\n")
        else:
            if not net_info.get("computed", False):
                lines.append(f"- Net-load not computed: {net_info.get('reason')}\n")
            else:
                lines.append("- Net-load computed\n")
                lines.append(f"- Rows: {net_info.get('n_rows')}\n")
                lines.append(f"- Artifact: `{net_info.get('path')}`\n")

        report_path.write_text("".join(lines), encoding="utf-8")
        return report_path.as_posix()

    def tick(self) -> None:
        for msg in self.poll(max_messages=200):
            if msg.kind != "result":
                continue
            if msg.payload.get("event") != "evaluation_complete":
                continue

            track = msg.payload.get("track")
            if track not in ("load", "solar"):
                continue

            self._results[track] = {
                "track": track,
                "model_name": msg.payload.get("model_name") or msg.payload.get("model_type"),
                "strategy": msg.payload.get("strategy", ""),
                "metrics": msg.payload.get("metrics", {}),
                "artifacts": msg.payload.get("artifacts", {}),
            }

        if "load" in self._results and "solar" in self._results and not self._results.get("_integrated"):
            net_info = self._try_compute_net_load()
            report_path = self._write_report(net_info)
            self._results["_integrated"] = True

            self.publish(
                topic_type="integrate",
                kind="result",
                payload={
                    "event": "joint_report_ready",
                    "report_path": report_path,
                    "net_load": net_info or {"computed": False, "reason": "missing tracks"},
                },
            )


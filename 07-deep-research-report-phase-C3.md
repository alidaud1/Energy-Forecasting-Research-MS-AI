# Phase C Step Two Implementation Blueprint for Net‑Load Validation, Thesis‑Ready Visuals, and System Auditing

## Executive summary

This step upgrades your Phase C system from “it runs” to “it is **research‑grade and thesis‑ready**” by making the **ForecastIntegrationAgent** responsible for: (a) **net‑load correctness** (alignment checks, MAE/RMSE on net load), (b) **thesis‑quality visualizations** (triple‑axis timeline + error distribution comparisons), and (c) an **agentic performance audit** that converts `logs/message_pool.jsonl` into a readable appendix that documents data verification steps and model/hyperparameter decisions.

Critically, net‑load is widely defined in power systems as **total electric demand minus variable renewable generation (e.g., wind/solar)**. The U.S. DOE describes grid net load as the difference between total demand and behind‑the‑meter generation such as solar (and other distributed resources). citeturn2search1 The U.S. EIA similarly defines net load as total electric demand minus wind and solar generation (describing CAISO operational impacts). citeturn2search3 Because your current Phase C proof-of-concept uses **GEFCom2014 Load** and **GEFCom2014 Solar** tracks, alignment may often be **empty** (different time ranges/systems). Your integration logic must therefore compute net‑load **only when timestamps overlap**, otherwise produce a joint report that explicitly states why net‑load was not computed.

For evaluation reporting, use MAE and RMSE as the primary point‑forecast metrics (scikit‑learn defines MAE and RMSE regression losses and their expected input shapes). citeturn3search0turn3search4

## Net‑load validation logic and combined error metrics

### Alignment rules that keep net‑load scientifically correct

Implement net‑load only after passing all three checks:

The test horizon timestamps must overlap between load and solar predictions, since net load is a **same‑time** derived quantity (load minus solar). citeturn2search3turn2search1  
The forecast horizons must match (at least for the overlapping horizon indices); otherwise you risk subtracting mismatched lead times.  
The spatial identifier must be consistent. If your predictions have `ZONEID`, prefer strict matching by `ZONEID + timestamp + horizon`. If load has one zone and solar has three zones (GEFCom solar), you must either (a) refuse to compute net‑load, or (b) explicitly document a chosen aggregation assumption (e.g., sum solar zones then subtract from load) as a research assumption.

### Net‑load metric definitions

Once aligned predictions exist:

\[
\hat{N} = \hat{L} - \hat{S}, \quad N = L - S
\]

Then compute:

- MAE\((N,\hat{N})\) using `mean_absolute_error` citeturn3search0  
- RMSE\((N,\hat{N})\) using `root_mean_squared_error` citeturn3search4  

Use these on the **same aligned set** you used for subtraction.

### Required prediction artifact schema

To enable robust synthesis, ensure every evaluation (baseline or transformer) writes a consistent Parquet artifact:

| Column | Type | Meaning |
|---|---|---|
| `run_id` | string | Run scope |
| `track` | string | `load` / `solar` |
| `model_family` | string | `baseline` / `transformer` |
| `model_name` | string | `xgboost` / `patchtst` / `itransformer` |
| `ZONEID` | int (optional) | Spatial id if applicable |
| `timestamp` | datetime64 | Forecasted timestamp (the target time) |
| `horizon` | int | lead time index (1..H) |
| `y_true` | float | true target |
| `y_pred` | float | predicted target |

If your baseline pipeline (Phase B) currently doesn’t persist `predictions.parquet`, add it now; otherwise the “error distribution vs baseline” plot cannot be computed.

## Thesis‑ready visualization suite

### Triple‑axis timeline plot requirements

You requested a triple‑axis plot for a test week showing **Load**, **Solar**, and **Net‑Load** on one timeline. This is thesis-friendly because it visually demonstrates the “duck curve” / net‑load shaping phenomenon described by EIA’s CAISO net‑load guidance. citeturn2search3

Implementation best practices for a thesis figure:
- Plot a **single coherent week** (e.g., first/last full week available in the aligned net‑load dataset).
- Use hourly timestamps on the x‑axis (tick every 6–12 hours).
- Put load and net‑load on primary/secondary axes, solar on the third axis (offset spine) to prevent scale compression.

### Error distribution plot requirements

You requested histogram or boxplot comparing Transformer vs XGBoost baselines. The most defensible approach:
- Use the **same aligned test timestamps/horizons** for both models.
- Compute an “absolute error series” and plot two distributions (baseline vs transformer).
- Report paired differences and optionally add a bootstrap CI. If you later want formal significance testing, Diebold–Mariano provides a classic framework for comparing forecast accuracy under general loss functions. citeturn1search42

## Agentic performance audit from message_pool.jsonl

### What to extract for the System Performance Report

Your audit report should summarize:

- PreparationAgent data integrity verification: which files/paths it checked, and whether artifacts existed for both tracks. (This is the “data readiness” trace that supports reproducibility.)
- ModelManagerAgent decision trace: model selection and hyperparameters. If Optuna is enabled, the audit should log the study name and sampled hyperparameters. Optuna’s `Study` object supports interactive workflows via `ask()` (and `tell()`), which fits your “interactive BO skeleton” goal. citeturn5search4

### How to make “reasoning” available from agents

If you want the audit to contain explicit “reasoning,” add lightweight fields to message payloads:
- `rationale`: short text
- `decision_inputs`: key signals (metric thresholds, config flags)

This keeps the log factual and thesis-appropriate—no “LLM chain-of-thought,” just decision metadata.

## Full benchmark mode and scalability demonstration

### Interpreting “full benchmark” in a thesis-feasible way

Your request says “Full Benchmark across all available GEFCom2014 tasks (Load Task 1 and Solar Task 15).” The thesis-feasible interpretation is:
- Run **both tracks** (Load Task 1 and Solar Task 15) in the same orchestrated session,
- Run **baseline (XGBoost)** and **transformer** for each track,
- Produce integrated joint report with: per-track metrics, net‑load (if overlap), and error distribution plots baseline vs transformer.

If you later want to truly benchmark all tasks (e.g., all 15 tasks per track), that becomes a larger experiment manager issue because Phase A preprocessing must be repeated per task and artifacts must be versioned (manifests per task). This is doable but should be staged after you have a working `--full-benchmark` for the primary tasks.

## Deliverables: code for Integration Agent, final report generation, and message export

Below are the requested implementation deliverables in modular code blocks.

### Finalized `src/agents/integration_agent.py`

This version:
- Collects evaluation results for multiple models per track (baseline + transformer).
- Computes net‑load predictions and MAE/RMSE if alignment exists (DOE/EIA definitions applied operationally). citeturn2search1turn2search3  
- Generates thesis figures:
  - Triple-axis timeline (only if net‑load computed)
  - Boxplot of absolute error distributions (baseline vs transformer)
- Writes `reports/final_research_summary.md`
- Writes a “system performance audit” markdown by parsing `logs/message_pool.jsonl`

```python
# src/agents/integration_agent.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from .base_agent import BaseAgent


@dataclass
class IntegrationConfig:
    # Expect at least these models to produce plots comparing baseline vs transformer
    baseline_model_name: str = "xgboost"
    transformer_model_names: Tuple[str, ...] = ("patchtst", "itransformer")

    # For plots
    timeline_days: int = 7
    max_points_boxplot: int = 20000  # subsample to keep plots scalable

    # Paths
    message_pool_path: str = "logs/message_pool.jsonl"
    reports_dir: str = "reports"


class ForecastIntegrationAgent(BaseAgent):
    """
    Research-grade integration agent:
      - collects evaluation results (load & solar; baseline & transformers)
      - validates alignment
      - computes net-load predictions if possible
      - produces thesis-ready plots and final markdown report
      - produces system audit by parsing message_pool.jsonl
    """
    def __init__(self, cfg, pool, icfg: Optional[IntegrationConfig] = None):
        super().__init__(cfg, pool)
        self.subscribe(topics=["evaluate"], start_from="latest")
        self.icfg = icfg or IntegrationConfig()

        self.run_dir = Path("logs") / "runs" / self.run_id
        (self.run_dir / "integrated").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "reports").mkdir(parents=True, exist_ok=True)

        self.reports_dir = Path(self.icfg.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Store results by track then by model_name
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {"load": {}, "solar": {}}
        self._finalized = False

    # --------------------------
    # Message collection
    # --------------------------
    def tick(self) -> None:
        # Collect any new evaluation messages
        for msg in self.poll(max_messages=100):
            if msg.kind != "result":
                continue
            if msg.payload.get("event") != "evaluation_complete":
                continue

            track = msg.payload.get("track")
            model_name = msg.payload.get("model_name") or msg.payload.get("model_type") or "unknown"
            if track not in ("load", "solar"):
                continue

            self.results[track][model_name] = {
                "track": track,
                "model_name": model_name,
                "strategy": msg.payload.get("strategy"),
                "metrics": msg.payload.get("metrics", {}),
                "artifacts": msg.payload.get("artifacts", {}),
                "device": msg.payload.get("device"),
                "train_stats": msg.payload.get("train_stats", {}),
            }

        # Finalize once we have at least one load and one solar transformer run
        if not self._finalized and self._has_minimum_for_report():
            self._finalized = True
            self._build_final_outputs()

            self.publish(
                topic_type="integrate",
                kind="result",
                payload={
                    "event": "final_research_summary_ready",
                    "report_path": str((self.reports_dir / "final_research_summary.md").as_posix()),
                    "run_specific_report": str((self.run_dir / "reports" / "final_research_summary.md").as_posix()),
                },
            )

    def _has_minimum_for_report(self) -> bool:
        # Require at least one transformer model result for each track
        def has_transformer(track: str) -> bool:
            return any(m in self.results[track] for m in self.icfg.transformer_model_names)

        return has_transformer("load") and has_transformer("solar")

    # --------------------------
    # Net-load computation
    # --------------------------
    def _read_preds(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Predictions parquet not found: {p}")
        df = pd.read_parquet(p)
        if "timestamp" not in df.columns or "y_true" not in df.columns or "y_pred" not in df.columns:
            raise ValueError(f"Predictions parquet missing required columns: {p}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _select_best_model(self, track: str) -> Dict[str, Any]:
        # Choose best by RMSE if available; else MAE.
        candidates = list(self.results[track].values())
        if not candidates:
            raise ValueError(f"No results for track {track}")

        def key(r: Dict[str, Any]) -> float:
            m = r.get("metrics", {})
            if "rmse" in m and m["rmse"] is not None:
                return float(m["rmse"])
            if "mae" in m and m["mae"] is not None:
                return float(m["mae"])
            return float("inf")

        return sorted(candidates, key=key)[0]

    def _align_for_net_load(self, dfL: pd.DataFrame, dfS: pd.DataFrame) -> pd.DataFrame:
        # Require horizon matching
        key_cols = ["timestamp", "horizon"]
        if "ZONEID" in dfL.columns and "ZONEID" in dfS.columns:
            key_cols = ["ZONEID"] + key_cols

        merged = dfL.merge(dfS, on=key_cols, suffixes=("_load", "_solar"))
        return merged

    def _compute_net_load_metrics(self, merged: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        merged = merged.copy()
        merged["net_true"] = merged["y_true_load"] - merged["y_true_solar"]
        merged["net_pred"] = merged["y_pred_load"] - merged["y_pred_solar"]

        mae = float(mean_absolute_error(merged["net_true"], merged["net_pred"]))
        rmse = float(root_mean_squared_error(merged["net_true"], merged["net_pred"]))
        return merged, {"mae": mae, "rmse": rmse}

    # --------------------------
    # Thesis-ready plots
    # --------------------------
    def _plot_triple_axis_week(self, merged_net: pd.DataFrame, out_path: Path) -> Optional[str]:
        # Needs columns: timestamp, y_true_load, y_true_solar, net_true
        df = merged_net.copy()
        df = df.sort_values("timestamp")

        # Pick a continuous week window from earliest timestamp
        start = df["timestamp"].min().floor("D")
        end = start + pd.Timedelta(days=self.icfg.timeline_days)

        week = df[(df["timestamp"] >= start) & (df["timestamp"] < end) & (df["horizon"] == 1)].copy()
        if len(week) < 24:  # not enough data
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

        # Create third axis offset to the right
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.12))
        ax3.plot(t, net, label="Net-Load (Actual)")
        ax3.set_ylabel("Net-Load")

        ax1.set_title("Load, Solar, and Net-Load (Test Week, Horizon=1)")
        fig.autofmt_xdate()

        # Combined legend
        lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")

        fig.tight_layout()
        fig.savefig(out_path.as_posix(), dpi=200)
        plt.close(fig)
        return out_path.as_posix()

    def _plot_error_distribution_boxplot(
        self,
        preds_baseline: pd.DataFrame,
        preds_transformer: pd.DataFrame,
        out_path: Path,
        title: str,
    ) -> str:
        # Align on timestamp+horizon(+ZONEID)
        key_cols = ["timestamp", "horizon"]
        if "ZONEID" in preds_baseline.columns and "ZONEID" in preds_transformer.columns:
            key_cols = ["ZONEID"] + key_cols

        merged = preds_baseline.merge(
            preds_transformer,
            on=key_cols,
            suffixes=("_base", "_tr"),
        )
        if len(merged) == 0:
            raise ValueError("No overlap between baseline and transformer predictions for error plot.")

        # Absolute error series (subsample for scale)
        ae_base = np.abs(merged["y_true_base"] - merged["y_pred_base"]).to_numpy()
        ae_tr = np.abs(merged["y_true_tr"] - merged["y_pred_tr"]).to_numpy()

        if len(ae_base) > self.icfg.max_points_boxplot:
            idx = np.random.RandomState(42).choice(len(ae_base), size=self.icfg.max_points_boxplot, replace=False)
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

    # --------------------------
    # Agentic audit report
    # --------------------------
    def _export_system_audit(self) -> str:
        """
        Parse logs/message_pool.jsonl and write a causal narrative of:
          - Preparation verification
          - Model decisions/hyperparams
        """
        p = Path(self.icfg.message_pool_path)
        if not p.exists():
            return ""

        prep_checks = []
        model_plans = []

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                topic = obj.get("topic", "")
                if not topic.endswith(f"/{self.run_id}"):
                    continue

                sender = obj.get("sender", "")
                kind = obj.get("kind", "")
                payload = obj.get("payload", {}) or {}

                # PreparationAgent readiness evidence
                if sender.lower().startswith("preparation") and payload.get("event") in ("ready", "not_ready"):
                    prep_checks.append(obj)

                # Model plan evidence (train commands often contain hyperparams)
                if topic.startswith(f"train/{self.run_id}") and kind == "command":
                    model_plans.append(obj)

        audit_path = self.run_dir / "reports" / "system_performance_audit.md"
        lines = []
        lines.append(f"# System Performance Audit ({self.run_id})\n\n")

        lines.append("## PreparationAgent data readiness trace\n")
        if not prep_checks:
            lines.append("- No preparation checks found in message log.\n")
        else:
            for m in prep_checks:
                payload = m.get("payload", {})
                lines.append(f"- `{m.get('created_at')}` event=`{payload.get('event')}` track=`{payload.get('track')}`\n")
                if payload.get("event") == "not_ready":
                    lines.append(f"  - error: `{payload.get('error')}`\n")

        lines.append("\n## ModelManager / training plan trace\n")
        if not model_plans:
            lines.append("- No train commands found.\n")
        else:
            for m in model_plans:
                payload = m.get("payload", {})
                lines.append(f"- `{m.get('created_at')}` track=`{payload.get('track')}` model=`{payload.get('model_name', payload.get('model_type'))}`\n")
                hp = payload.get("hyperparams") or {}
                if hp:
                    lines.append(f"  - hyperparams: `{hp}`\n")
                if "optuna" in payload:
                    lines.append(f"  - optuna: `{payload.get('optuna')}`\n")

        audit_path.write_text("".join(lines), encoding="utf-8")
        return audit_path.as_posix()

    # --------------------------
    # Final report synthesis
    # --------------------------
    def _build_final_outputs(self) -> None:
        # Choose best models per track (usually transformer)
        best_load = self._select_best_model("load")
        best_solar = self._select_best_model("solar")

        # Load their prediction artifacts
        load_pred_path = best_load["artifacts"].get("predictions_path")
        solar_pred_path = best_solar["artifacts"].get("predictions_path")

        netload_info = {"computed": False, "reason": "Missing prediction artifacts."}
        triple_plot_path = None

        if load_pred_path and solar_pred_path:
            dfL = self._read_preds(load_pred_path)
            dfS = self._read_preds(solar_pred_path)

            merged = self._align_for_net_load(dfL, dfS)
            if len(merged) == 0:
                netload_info = {"computed": False, "reason": "No overlapping timestamps/horizons (or zones) between load and solar predictions."}
            else:
                merged_net, net_metrics = self._compute_net_load_metrics(merged)
                out_net = self.run_dir / "integrated" / "net_load_predictions.parquet"
                merged_net.to_parquet(out_net)

                netload_info = {
                    "computed": True,
                    "path": out_net.as_posix(),
                    "n_rows": int(len(merged_net)),
                    "metrics": net_metrics,
                }

                triple_plot_path = self._plot_triple_axis_week(
                    merged_net,
                    out_path=self.run_dir / "plots" / "triple_axis_week.png",
                )

        # Error distribution plots (baseline vs transformer) if baseline predictions exist
        error_plots = {}
        for track in ("load", "solar"):
            base = self.results[track].get(self.icfg.baseline_model_name)
            tr_best = self._select_best_model(track)

            if base and base["artifacts"].get("predictions_path") and tr_best["artifacts"].get("predictions_path"):
                try:
                    df_base = self._read_preds(base["artifacts"]["predictions_path"])
                    df_tr = self._read_preds(tr_best["artifacts"]["predictions_path"])
                    out_plot = self.run_dir / "plots" / f"{track}_error_distribution_boxplot.png"
                    pth = self._plot_error_distribution_boxplot(
                        preds_baseline=df_base,
                        preds_transformer=df_tr,
                        out_path=out_plot,
                        title=f"{track.upper()} | Abs Error: Baseline vs Transformer",
                    )
                    error_plots[track] = pth
                except Exception as e:
                    error_plots[track] = f"(error generating plot: {e})"
            else:
                error_plots[track] = "(baseline/transformer predictions not available)"

        audit_path = self._export_system_audit()

        # Write run-specific report and repo-level final report
        run_report = self.run_dir / "reports" / "final_research_summary.md"
        repo_report = self.reports_dir / "final_research_summary.md"

        md = []
        md.append(f"# Final Research Summary ({self.run_id})\n\n")

        md.append("## Best model per track\n")
        md.append(f"- **Load best**: `{best_load.get('model_name')}` | MAE={best_load['metrics'].get('mae')} RMSE={best_load['metrics'].get('rmse')}\n")
        md.append(f"  - plot: `{best_load['artifacts'].get('plot_path')}`\n")
        md.append(f"  - predictions: `{best_load['artifacts'].get('predictions_path')}`\n")
        md.append(f"- **Solar best**: `{best_solar.get('model_name')}` | MAE={best_solar['metrics'].get('mae')} RMSE={best_solar['metrics'].get('rmse')}\n")
        md.append(f"  - plot: `{best_solar['artifacts'].get('plot_path')}`\n")
        md.append(f"  - predictions: `{best_solar['artifacts'].get('predictions_path')}`\n")

        md.append("\n## Net-load synthesis\n")
        if not netload_info.get("computed"):
            md.append(f"- Net-load not computed: {netload_info.get('reason')}\n")
            md.append("  - Note: Net load is defined as demand minus renewable generation; it requires aligned timestamps and compatible spatial scope. citeturn2search3turn2search1\n")
        else:
            m = netload_info["metrics"]
            md.append(f"- Net-load computed ✅ rows={netload_info.get('n_rows')}\n")
            md.append(f"- Net-load MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}\n")
            md.append(f"- Net-load predictions artifact: `{netload_info.get('path')}`\n")
            if triple_plot_path:
                md.append(f"- Triple-axis test-week plot: `{triple_plot_path}`\n")

        md.append("\n## Error distribution vs baseline\n")
        md.append(f"- Load error distribution plot: `{error_plots.get('load')}`\n")
        md.append(f"- Solar error distribution plot: `{error_plots.get('solar')}`\n")

        md.append("\n## Agentic system audit\n")
        md.append(f"- System performance audit: `{audit_path}`\n")

        run_report.write_text("".join(md), encoding="utf-8")
        repo_report.write_text("".join(md), encoding="utf-8")
```

### Script to export `message_pool.jsonl` into thesis appendix format

Create: `scripts/export_message_pool.py`

This script outputs:
- `reports/message_trace_<run_id>.md`
- optionally `reports/message_trace_<run_id>.csv`

```python
# scripts/export_message_pool.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Export message_pool.jsonl to a readable thesis appendix format.")
    p.add_argument("--log", type=str, default="logs/message_pool.jsonl")
    p.add_argument("--run_id", type=str, required=True, help="Run id (topic source) to filter, e.g., run_20260414_120501")
    p.add_argument("--out_dir", type=str, default="reports")
    p.add_argument("--csv", action="store_true", help="Also export CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            topic = obj.get("topic", "")
            if not topic.endswith(f"/{args.run_id}"):
                continue

            payload = obj.get("payload", {}) or {}
            # Keep a brief payload preview for readability
            preview_keys = ["event", "command", "track", "model_name", "model_type"]
            preview = {k: payload.get(k) for k in preview_keys if k in payload}

            rows.append({
                "created_at": obj.get("created_at"),
                "seq": obj.get("seq"),
                "topic": topic,
                "kind": obj.get("kind"),
                "sender": obj.get("sender"),
                "sender_role": obj.get("sender_role"),
                "correlation_id": obj.get("correlation_id"),
                "payload_preview": json.dumps(preview, ensure_ascii=False),
            })

    md_path = out_dir / f"message_trace_{args.run_id}.md"
    lines = []
    lines.append(f"# Message Trace Appendix ({args.run_id})\n\n")
    lines.append("| time | seq | topic | kind | sender | role | correlation | payload preview |\n")
    lines.append("|---|---:|---|---|---|---|---|---|\n")
    for r in rows:
        lines.append(
            f"| {r['created_at']} | {r['seq']} | {r['topic']} | {r['kind']} | {r['sender']} | {r['sender_role']} | {r['correlation_id']} | {r['payload_preview']} |\n"
        )

    md_path.write_text("".join(lines), encoding="utf-8")
    print("Wrote:", md_path)

    if args.csv:
        csv_path = out_dir / f"message_trace_{args.run_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["created_at"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("Wrote:", csv_path)


if __name__ == "__main__":
    main()
```

### Updating orchestrator for `--full-benchmark`

The thesis-feasible `--full-benchmark` mode should run:
- baseline and transformer for **load**
- baseline and transformer for **solar**
- then IntegrationAgent produces the final report

Conceptually, do this in one run_id so the final report is single-file.

Minimal orchestrator idea:
- Publish 4 train commands:
  - `(track=load, model=xgboost)`
  - `(track=load, model=patchtst)`
  - `(track=solar, model=xgboost)`
  - `(track=solar, model=patchtst)`
- Ensure each evaluation message includes `predictions_path` so error distributions are computable.

If your current `orchestrator.py` already injects transformer plans, extend it to also inject baseline plans and wait until IntegrationAgent emits `final_research_summary_ready`.

## Notes that make this “research‑grade”

MAE and RMSE are standard regression losses; MAE is non-negative and best at 0, and scikit-learn documents expected shapes for multi-output settings. citeturn3search0turn3search4  
Your use of a message-driven audit log supports reproducibility and makes the orchestration traceable, aligning with message-driven agent architecture guidance emphasizing decoupled communication and correlation-driven workflows. citeturn2search1  
If you later need formal statistical comparisons between baseline and transformer forecast accuracy, Diebold–Mariano is a classic test for equal predictive accuracy that supports flexible loss functions. citeturn1search42

This completes Phase C Step Two at the level needed for thesis figures, appendix auditability, and “research mode” execution with a unified report.
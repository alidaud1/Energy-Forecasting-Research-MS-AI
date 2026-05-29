"""
Run baseline models (base-paper set) for GEFCom2014 Load Task 1..15 and collect metrics.

This script:
1) Assumes Phase-A manifests already exist under data/processed/load_tasks/task_XX/dataset_manifest.json
   (use scripts/prepare_load_all_tasks.py).
2) Runs orchestrator baseline mode for each task+model.
3) Extracts MAE/RMSE from logs/message_pool.jsonl for each run_id and writes a CSV summary.

Run inside the project venv on Windows:
  (venv) python scripts/run_load_all_tasks_baselines.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


BASE_PAPER_MODELS = ["linear", "svr", "mlp", "lstm", "gru", "cnn", "xgboost"]


def _parse_run_id(stdout: str) -> str:
    # Looks for: [START] mode=baseline run_id=run_YYYY... track=load model=...
    for line in stdout.splitlines():
        if "run_id=" in line and "[START]" in line:
            # naive parse
            parts = line.split()
            for p in parts:
                if p.startswith("run_id="):
                    return p.split("=", 1)[1].strip()
    raise ValueError("Could not parse run_id from orchestrator output.")


def _load_metrics_from_message_pool(message_pool: Path, run_id: str) -> Tuple[float, float]:
    # Find the last evaluate:result for this run_id.
    last: Optional[Dict] = None
    with message_pool.open("r", encoding="utf-8") as f:
        for line in f:
            if run_id not in line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if msg.get("run_id") != run_id:
                continue
            if msg.get("topic", "").startswith("evaluate/") and msg.get("kind") == "result":
                payload = msg.get("payload") or {}
                if payload.get("event") == "evaluation_complete":
                    last = msg
    if not last:
        raise ValueError(f"No evaluate:result found for run_id={run_id}")
    metrics = (last.get("payload") or {}).get("metrics") or {}
    return float(metrics["mae"]), float(metrics["rmse"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_root", type=str, default="data/processed/load_tasks", help="Root with task_XX manifests.")
    ap.add_argument("--tasks", type=str, default="1-15", help="Task range, e.g. 1-15 or 1,2,3.")
    ap.add_argument("--models", type=str, default=",".join(BASE_PAPER_MODELS), help="Comma-separated baseline models.")
    ap.add_argument("--timeout_seconds", type=float, default=7200.0)
    ap.add_argument("--tick_seconds", type=float, default=1.0)
    ap.add_argument("--max_train_windows", type=int, default=20000)
    ap.add_argument("--max_test_windows", type=int, default=5000)
    ap.add_argument("--W", type=int, default=168)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--out_csv", type=str, default="logs/load_all_tasks_baselines.csv")
    ap.add_argument("--message_pool", type=str, default="logs/message_pool.jsonl")
    ap.add_argument("--no_precheck", action="store_true", help="Pass --no_precheck to orchestrator (skip Phase C precheck).")
    args = ap.parse_args()

    tasks_root = Path(args.tasks_root)
    message_pool = Path(args.message_pool)

    # tasks
    t = str(args.tasks).strip()
    if "-" in t:
        a, b = t.split("-", 1)
        task_ids = list(range(int(a), int(b) + 1))
    else:
        task_ids = [int(x.strip()) for x in t.split(",") if x.strip()]

    models = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]
    if not models:
        raise SystemExit("No models selected.")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    rows = []

    for task_id in task_ids:
        manifest = tasks_root / f"task_{task_id:02d}" / "dataset_manifest.json"
        if not manifest.exists():
            raise SystemExit(f"Missing manifest for task {task_id}: {manifest}")

        for model in models:
            print(f"[RUN] task={task_id:02d} model={model}", flush=True)
            cmd = [
                py,
                "orchestrator.py",
                "--mode",
                "baseline",
                "--manifest",
                str(manifest.as_posix()),
                "--track",
                "load",
                "--model",
                model,
                "--timeout_seconds",
                str(args.timeout_seconds),
                "--tick_seconds",
                str(args.tick_seconds),
                "--max_train_windows",
                str(args.max_train_windows),
                "--max_test_windows",
                str(args.max_test_windows),
                "--W",
                str(args.W),
                "--H",
                str(args.H),
                "--stride",
                str(args.stride),
            ]
            if args.no_precheck:
                cmd.append("--no_precheck")
            # Capture output to parse run_id
            try:
                p = subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                out = (e.stdout or "") + "\n" + (e.stderr or "")
                print("[ERROR] orchestrator failed. task=%02d model=%s" % (task_id, model), flush=True)
                print(out.strip()[:4000], flush=True)
                raise
            stdout = (p.stdout or "") + "\n" + (p.stderr or "")
            run_id = _parse_run_id(stdout)
            # allow flush-to-jsonl delay
            time.sleep(0.2)
            mae, rmse = _load_metrics_from_message_pool(message_pool, run_id)
            rows.append({"task_id": task_id, "model": model, "run_id": run_id, "mae": mae, "rmse": rmse})
            print(f"[OK] run_id={run_id} mae={mae:.4f} rmse={rmse:.4f}", flush=True)

    # Write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "model", "run_id", "mae", "rmse"])
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] Wrote {out_csv} ({len(rows)} rows).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

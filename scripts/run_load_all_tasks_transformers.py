"""
Run transformer models (PatchTST / iTransformer) for GEFCom2014 Load Tasks 1..15.

Requires per-task manifests under data/processed/load_tasks/task_XX/dataset_manifest.json.

Example:
  (venv) python scripts/run_load_all_tasks_transformers.py --model patchtst --epochs 5
  (venv) python scripts/run_load_all_tasks_transformers.py --model itransformer --epochs 5

Writes:
  logs/load_all_tasks_transformer_<model>.csv
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


def _parse_run_id(stdout: str) -> str:
    for line in stdout.splitlines():
        if "run_id=" in line and "[START]" in line:
            # [START] mode=research run_id=run_... load_model=...
            for part in line.split():
                if part.startswith("run_id="):
                    return part.split("=", 1)[1].strip()
    raise ValueError("Could not parse run_id from orchestrator output.")


def _final_eval_metrics(message_pool: Path, run_id: str, *, track: str = "load") -> Tuple[float, float]:
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
            if not str(msg.get("topic") or "").startswith("evaluate/"):
                continue
            if msg.get("kind") != "result":
                continue
            payload = msg.get("payload") or {}
            if payload.get("event") != "evaluation_complete":
                continue
            if str(payload.get("track") or "") != track:
                continue
            last = msg
    if not last:
        raise ValueError(f"No evaluation_complete result for run_id={run_id} track={track}")
    metrics = (last.get("payload") or {}).get("metrics") or {}
    return float(metrics["mae"]), float(metrics["rmse"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_root", type=str, default="data/processed/load_tasks")
    ap.add_argument("--tasks", type=str, default="1-15")
    ap.add_argument("--model", type=str, required=True, choices=["patchtst", "itransformer"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--timeout_seconds", type=float, default=21600.0)
    ap.add_argument("--max_train_windows", type=int, default=20000)
    ap.add_argument("--max_test_windows", type=int, default=5000)
    ap.add_argument("--W", type=int, default=168)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--message_pool", type=str, default="logs/message_pool.jsonl")
    ap.add_argument("--out_csv", type=str, default="", help="Default: logs/load_all_tasks_transformer_<model>.csv")
    ap.add_argument("--no_precheck", action="store_true")
    args = ap.parse_args()

    tasks_root = Path(args.tasks_root)
    message_pool = Path(args.message_pool)
    out_csv = Path(args.out_csv) if args.out_csv else Path(f"logs/load_all_tasks_transformer_{args.model}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    t = str(args.tasks).strip()
    if "-" in t:
        a, b = t.split("-", 1)
        task_ids = list(range(int(a), int(b) + 1))
    else:
        task_ids = [int(x.strip()) for x in t.split(",") if x.strip()]

    py = sys.executable
    rows = []
    for task_id in task_ids:
        manifest = tasks_root / f"task_{task_id:02d}" / "dataset_manifest.json"
        if not manifest.exists():
            raise SystemExit(f"Missing manifest for task {task_id}: {manifest}")

        print(f"[RUN] transformer task={task_id:02d} model={args.model}", flush=True)
        cmd = [
            py,
            "orchestrator.py",
            "--mode",
            "research",
            "--manifest",
            str(manifest.as_posix()),
            "--research_tracks",
            "load",
            "--load_model",
            str(args.model),
            "--epochs",
            str(int(args.epochs)),
            "--batch_size",
            str(int(args.batch_size)),
            "--lr",
            str(float(args.lr)),
            "--patience",
            str(int(args.patience)),
            "--timeout_seconds",
            str(float(args.timeout_seconds)),
            "--max_train_windows",
            str(int(args.max_train_windows)),
            "--max_test_windows",
            str(int(args.max_test_windows)),
            "--W",
            str(int(args.W)),
            "--H",
            str(int(args.H)),
            "--stride",
            str(int(args.stride)),
        ]
        if args.no_precheck:
            cmd.append("--no_precheck")

        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stdout = (p.stdout or "") + "\n" + (p.stderr or "")
        run_id = _parse_run_id(stdout)
        time.sleep(0.5)
        mae, rmse = _final_eval_metrics(message_pool, run_id, track="load")
        rows.append({"task_id": task_id, "model": args.model, "run_id": run_id, "mae": mae, "rmse": rmse})
        print(f"[OK] run_id={run_id} mae={mae:.4f} rmse={rmse:.4f}", flush=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "model", "run_id", "mae", "rmse"])
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] Wrote {out_csv} ({len(rows)} rows).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
Run HPO (Optuna ask/tell) for a single baseline model across GEFCom2014 load Tasks 1..15.

Default use case: XGBoost HPO across all tasks to match the base paper's Table-II setup.

This script:
1) Requires per-task manifests under data/processed/load_tasks/task_XX/dataset_manifest.json
   (use scripts/prepare_load_all_tasks.py).
2) Runs orchestrator baseline mode with --hpo --n_trials N for each task.
3) Extracts the best (min RMSE) evaluation_complete message for that run_id and writes a CSV.

Run inside the project venv on Windows:
  (venv) python scripts/run_load_all_tasks_hpo.py --model xgboost --n_trials 15
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
            for part in line.split():
                if part.startswith("run_id="):
                    return part.split("=", 1)[1].strip()
    raise ValueError("Could not parse run_id from orchestrator output.")


def _best_eval_from_message_pool(message_pool: Path, run_id: str) -> Dict[str, object]:
    """
    Prefer the final best-trial test evaluation (variant=hpo_best, optuna disabled).
    Fallback to the best optuna-enabled trial metric if the final evaluation is missing.
    """
    final_best: Optional[dict] = None
    best_trial: Optional[dict] = None
    best_rmse = float("inf")
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
            metrics = payload.get("metrics") or {}
            try:
                rmse = float(metrics.get("rmse"))
                mae = float(metrics.get("mae"))
            except Exception:
                continue
            opt = payload.get("optuna") or {}
            variant = str(payload.get("variant") or "")
            if variant == "hpo_best" and not bool(opt.get("enabled")):
                final_best = {
                    "mae": mae,
                    "rmse": rmse,
                    "hyperparams": payload.get("hyperparams") or {},
                    "optuna": opt,
                    "child_run_id": payload.get("child_run_id") or "",
                    "variant": variant,
                }
                # don't break; keep last in case multiple
                continue
            if bool(opt.get("enabled")) and rmse < best_rmse:
                best_rmse = rmse
                best_trial = {
                    "mae": mae,
                    "rmse": rmse,
                    "hyperparams": payload.get("hyperparams") or {},
                    "optuna": opt,
                    "child_run_id": payload.get("child_run_id") or "",
                    "variant": variant or "hpo_trial",
                }
    if final_best:
        return final_best
    if best_trial:
        return best_trial
    raise ValueError(f"No evaluation_complete results found for run_id={run_id}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_root", type=str, default="data/processed/load_tasks")
    ap.add_argument("--tasks", type=str, default="1-15")
    ap.add_argument("--model", type=str, default="xgboost", choices=["linear", "svr", "mlp", "lstm", "gru", "cnn", "xgboost"])
    ap.add_argument("--n_trials", type=int, default=15)
    ap.add_argument("--timeout_seconds", type=float, default=21600.0)
    ap.add_argument("--tick_seconds", type=float, default=1.0)
    ap.add_argument("--max_train_windows", type=int, default=20000)
    ap.add_argument("--max_test_windows", type=int, default=5000)
    ap.add_argument("--W", type=int, default=168)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--out_csv", type=str, default="", help="Output CSV path. Default is logs/load_all_tasks_hpo_<model>.csv")
    ap.add_argument("--message_pool", type=str, default="logs/message_pool.jsonl")
    ap.add_argument("--no_precheck", action="store_true")
    args = ap.parse_args()

    tasks_root = Path(args.tasks_root)
    message_pool = Path(args.message_pool)

    t = str(args.tasks).strip()
    if "-" in t:
        a, b = t.split("-", 1)
        task_ids = list(range(int(a), int(b) + 1))
    else:
        task_ids = [int(x.strip()) for x in t.split(",") if x.strip()]
    if not task_ids:
        raise SystemExit("No tasks selected.")

    out_csv = Path(args.out_csv) if args.out_csv else Path(f"logs/load_all_tasks_hpo_{args.model}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    rows = []

    for task_id in task_ids:
        manifest = tasks_root / f"task_{task_id:02d}" / "dataset_manifest.json"
        if not manifest.exists():
            raise SystemExit(f"Missing manifest for task {task_id}: {manifest}")

        print(f"[HPO] task={task_id:02d} model={args.model} n_trials={args.n_trials}", flush=True)
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
            str(args.model),
            "--hpo",
            "--n_trials",
            str(int(args.n_trials)),
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

        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stdout = (p.stdout or "") + "\n" + (p.stderr or "")
        run_id = _parse_run_id(stdout)

        # allow log flush
        time.sleep(0.5)
        best = _best_eval_from_message_pool(message_pool, run_id)

        rows.append(
            {
                "task_id": task_id,
                "model": args.model,
                "run_id": run_id,
                "best_mae": float(best["mae"]),
                "best_rmse": float(best["rmse"]),
                "best_hyperparams_json": json.dumps(best.get("hyperparams") or {}, ensure_ascii=False),
                "best_optuna_json": json.dumps(best.get("optuna") or {}, ensure_ascii=False),
            }
        )
        print(f"[OK] run_id={run_id} best_rmse={rows[-1]['best_rmse']:.4f} best_mae={rows[-1]['best_mae']:.4f}", flush=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "model",
                "run_id",
                "best_mae",
                "best_rmse",
                "best_hyperparams_json",
                "best_optuna_json",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] Wrote {out_csv} ({len(rows)} rows).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

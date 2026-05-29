"""
Prepare GEFCom2014 *load* artifacts for Task 1..15 (Phase A), writing a separate
manifest per task so baseline runs can be executed independently and aggregated.

This script is meant to be executed inside the project's Python venv on Windows:

  (venv) python scripts/prepare_load_all_tasks.py --load_root ..\\dataset\\GEFCom2014-L_V2

It will write:
  data/processed/load_tasks/task_01/dataset_manifest.json
  data/processed/load_tasks/task_02/dataset_manifest.json
  ...

Each task directory also contains load/{clean,train,val,test,train_scaled,...}.parquet and scaler.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--load_root",
        type=str,
        required=True,
        help="Path to GEFCom2014-L_V2 folder (contains Load/Task k/*.csv).",
    )
    ap.add_argument("--out_root", type=str, default="data/processed/load_tasks", help="Output root directory.")
    ap.add_argument("--tasks", type=str, default="1-15", help="Task range, e.g. 1-15 or 1,2,3.")
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.20)
    ap.add_argument(
        "--split_policy",
        type=str,
        default="min_windows",
        choices=["fixed", "min_windows"],
        help="Split policy. fixed uses ratios as provided. min_windows ensures val/test are large enough to form at least one window.",
    )
    ap.add_argument("--scaler_kind", type=str, default="standard", choices=["standard", "minmax"])
    ap.add_argument("--no_time_features", action="store_true")
    ap.add_argument("--W", type=int, default=168, help="Window history length (hours) used for min_windows split policy.")
    ap.add_argument("--H", type=int, default=24, help="Forecast horizon (hours) used for min_windows split policy.")
    args = ap.parse_args()

    load_root = Path(args.load_root)
    if not load_root.exists():
        raise SystemExit(f"--load_root not found: {load_root}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Parse tasks
    task_ids: list[int] = []
    t = str(args.tasks).strip()
    if "-" in t:
        a, b = t.split("-", 1)
        task_ids = list(range(int(a), int(b) + 1))
    else:
        task_ids = [int(x.strip()) for x in t.split(",") if x.strip()]
    if not task_ids:
        raise SystemExit("No tasks selected.")

    py = sys.executable
    for k in task_ids:
        task_dir = out_root / f"task_{k:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = task_dir / "dataset_manifest.json"
        print(f"[PHASE-A] Preparing load task {k} -> {manifest_path}", flush=True)

        train_ratio = float(args.train_ratio)
        val_ratio = float(args.val_ratio)
        test_ratio = float(args.test_ratio)
        # Many GEFCom tasks contain only ~3 weeks of labeled LOAD; with W=168,H=24, the test split
        # must be >= 192 hours to produce at least one evaluation window. This policy keeps ratios
        # similar but guarantees feasibility for window-based evaluation.
        if args.split_policy == "min_windows":
            min_holdout = int(args.W) + int(args.H)
            # Many load tasks are short. Ensure BOTH val and test have >= (W+H) hours
            # so SlidingWindowDataset can build at least one window for each.
            csv_path = load_root / "Load" / f"Task {k}" / f"L{k}-train.csv"
            n_rows = 0
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    v = row.get("LOAD", "")
                    if v and str(v).strip() and str(v).strip().lower() != "nan":
                        n_rows += 1
            if n_rows <= (min_holdout * 3):
                # Minimal viable: train, val, test each at least min_holdout.
                # Allocate remaining rows to train.
                min_val = min_holdout
                min_test = min_holdout
                min_train = min_holdout
                if n_rows < (min_train + min_val + min_test):
                    raise SystemExit(
                        f"Task {k} too short for W+H={min_holdout}: labeled_rows={n_rows}."
                    )
                # Convert mins to ratios; give extra to train.
                rem = n_rows - (min_val + min_test)
                train_ratio = rem / n_rows
                val_ratio = min_val / n_rows
                test_ratio = min_test / n_rows
            else:
                # For longer tasks, keep roughly the user's intent but enforce minimums.
                # Start from desired ratios.
                train_ratio = float(args.train_ratio)
                val_ratio = float(args.val_ratio)
                test_ratio = float(args.test_ratio)

                # Increase val/test if their implied lengths are below min_holdout.
                val_ratio = max(val_ratio, min_holdout / n_rows)
                test_ratio = max(test_ratio, min_holdout / n_rows)
                train_ratio = 1.0 - val_ratio - test_ratio
                if train_ratio <= 0:
                    # fallback to 1/3 each
                    train_ratio = 1.0 - 2.0 * (min_holdout / n_rows)
                    val_ratio = min_holdout / n_rows
                    test_ratio = min_holdout / n_rows

        cmd = [
            py,
            "-m",
            "src.prepare_data",
            "--tracks",
            "load",
            "--load_zip",
            str(load_root),
            "--out_dir",
            str(task_dir.as_posix()),
            "--load_task_id",
            str(k),
            "--train_ratio",
            str(train_ratio),
            "--val_ratio",
            str(val_ratio),
            "--test_ratio",
            str(test_ratio),
            "--scaler_kind",
            str(args.scaler_kind),
        ]
        if args.no_time_features:
            cmd.append("--no_add_time_features")
        subprocess.run(cmd, check=True)

    print("[OK] Prepared all requested load tasks.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

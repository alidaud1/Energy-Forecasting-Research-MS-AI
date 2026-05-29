"""
Generate base-paper-style figures for our system:

- Fig. 7 equivalent: Predicted vs Actual vs Adjusted (Load).
- Fig. 8 equivalent: Cumulative estimated token/message cost by stage/agent.

This script is intended to be run in the project's Python venv on Windows
where pandas/pyarrow/matplotlib are installed (the same environment used to
read Parquet during experiments).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _require_imports() -> Tuple[object, object, object]:
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing deps. Run inside your project venv with pandas/numpy/matplotlib installed."
        ) from e
    return pd, np, plt


def _load_predictions_parquet(pd, path: Path):
    df = pd.read_parquet(path)
    required = {"timestamp", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "ZONEID" in df.columns:
        df["ZONEID"] = df["ZONEID"].astype(int)
    if "horizon" in df.columns:
        df["horizon"] = df["horizon"].astype(int)
    return df


def _select_series(df, *, zone: int | None, horizon: int = 1, col: str = "y_pred"):
    d = df
    if "horizon" in d.columns:
        d = d[d["horizon"] == horizon]
    if zone is not None and "ZONEID" in d.columns:
        d = d[d["ZONEID"] == zone]
    d = d.sort_values("timestamp")
    return d[["timestamp", col]]


def make_fig7_pred_actual_adjusted(
    *,
    original_pred_path: Path,
    adjusted_pred_path: Path,
    out_path: Path,
    zone: int | None,
    horizon: int,
):
    pd, _np, plt = _require_imports()

    df0 = _load_predictions_parquet(pd, original_pred_path)
    df1 = _load_predictions_parquet(pd, adjusted_pred_path)

    # y_true is identical for original vs adjusted; take from original.
    y_true = _select_series(df0, zone=zone, horizon=horizon, col="y_true").rename(columns={"y_true": "Actual"})
    y_pred = _select_series(df0, zone=zone, horizon=horizon, col="y_pred").rename(columns={"y_pred": "Predicted"})
    y_adj = _select_series(df1, zone=zone, horizon=horizon, col="y_pred").rename(columns={"y_pred": "Adjusted"})

    merged = y_true.merge(y_pred, on="timestamp", how="inner").merge(y_adj, on="timestamp", how="inner")

    fig = plt.figure(figsize=(7.0, 3.2), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(merged["timestamp"], merged["Actual"], label="Actual", linewidth=1.0)
    ax.plot(merged["timestamp"], merged["Predicted"], label="Predicted", linewidth=1.0)
    ax.plot(merged["timestamp"], merged["Adjusted"], label="Adjusted", linewidth=1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (MW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), bbox_inches="tight")
    plt.close(fig)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_fig8_cumulative_token_proxy(
    *,
    message_pool_jsonl: Path,
    run_id: str,
    out_path: Path,
    char_to_token_ratio: float = 4.0,
):
    pd, _np, plt = _require_imports()

    # Stage is derived from topic prefix: prepare/train/evaluate/integrate/deploy/system...
    stage_order = ["prepare", "train", "evaluate", "integrate", "deploy", "system"]

    rows: List[Dict[str, object]] = []
    for msg in _iter_jsonl(message_pool_jsonl):
        if msg.get("run_id") != run_id:
            continue
        topic = str(msg.get("topic") or "")
        stage = topic.split("/", 1)[0] if "/" in topic else topic
        sender = str(msg.get("sender") or "unknown")
        payload = msg.get("payload")
        # Proxy: count characters in a normalized JSON payload (stable, reproducible).
        payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        est_tokens = max(1, int(len(payload_str) / max(char_to_token_ratio, 1e-6)))
        rows.append({"stage": stage, "sender": sender, "est_tokens": est_tokens})

    if not rows:
        raise ValueError(f"No messages found for run_id={run_id} in {message_pool_jsonl}")

    df = pd.DataFrame(rows)
    # Keep only known stages, but don't drop unknowns silently.
    df["stage"] = df["stage"].where(df["stage"].isin(stage_order), other="other")

    # Aggregate and plot stacked bars by sender across stages.
    pivot = (
        df.groupby(["stage", "sender"], as_index=False)["est_tokens"]
        .sum()
        .pivot(index="stage", columns="sender", values="est_tokens")
        .fillna(0)
    )
    # Ensure stage order.
    idx = [s for s in stage_order if s in pivot.index] + ([s for s in pivot.index if s not in stage_order])
    pivot = pivot.reindex(idx)

    fig = plt.figure(figsize=(7.0, 3.2), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    bottom = None
    for col in pivot.columns:
        vals = pivot[col].to_numpy()
        ax.bar(pivot.index.tolist(), vals, bottom=bottom, label=col)
        bottom = vals if bottom is None else bottom + vals
    ax.set_xlabel("Stage")
    ax.set_ylabel("Estimated tokens (proxy)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=7, frameon=True, ncol=2)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_pred", type=str, required=True, help="Path to original predictions parquet (load).")
    ap.add_argument("--adjusted_pred", type=str, required=True, help="Path to adjusted predictions parquet (load).")
    ap.add_argument("--run_id", type=str, required=True, help="Run id to filter message_pool.jsonl for Fig.8 proxy.")
    ap.add_argument("--message_pool", type=str, default="logs/message_pool.jsonl", help="Path to message_pool.jsonl.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for generated images.")
    ap.add_argument("--zone", type=int, default=1, help="ZONEID to plot for Fig.7 (load). Use -1 for all zones.")
    ap.add_argument("--horizon", type=int, default=1, help="Horizon step to plot (default: 1).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    zone = None if args.zone == -1 else int(args.zone)

    make_fig7_pred_actual_adjusted(
        original_pred_path=Path(args.original_pred),
        adjusted_pred_path=Path(args.adjusted_pred),
        out_path=out_dir / "fig7_load_pred_actual_adjusted.png",
        zone=zone,
        horizon=int(args.horizon),
    )

    make_fig8_cumulative_token_proxy(
        message_pool_jsonl=Path(args.message_pool),
        run_id=str(args.run_id),
        out_path=out_dir / "fig8_cumulative_token_proxy.png",
    )

    print("OK: wrote fig7_load_pred_actual_adjusted.png and fig8_cumulative_token_proxy.png to", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


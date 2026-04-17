from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrecheckIssue:
    track: str
    split: str
    kind: str
    detail: str


def _hours_inclusive(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start) / pd.Timedelta(hours=1)) + 1


def _gap_hours_total(ts: pd.DatetimeIndex) -> int:
    if len(ts) == 0:
        return 0
    expected = _hours_inclusive(pd.Timestamp(ts.min()), pd.Timestamp(ts.max()))
    return int(expected - len(ts))


def _non_1h_steps(ts: pd.DatetimeIndex) -> int:
    if len(ts) <= 1:
        return 0
    diffs = ts.to_series().diff().dropna()
    return int((diffs != pd.Timedelta(hours=1)).sum())


def _na_sum(df: pd.DataFrame, cols: List[str]) -> int:
    if not cols:
        return 0
    return int(df[cols].isna().sum().sum())


def _read_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_path(manifest_path: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    # Prefer manifest-relative. Fall back to cwd-relative.
    cand1 = (manifest_path.parent / p).resolve()
    if cand1.exists():
        return cand1
    return (Path.cwd() / p).resolve()


def check_track_splits(
    *,
    track: str,
    manifest_path: Path,
    manifest: Dict[str, Any],
    check_fractional_weather: bool = True,
) -> Tuple[bool, List[PrecheckIssue], Dict[str, Any]]:
    """
    Validates Phase A2 outputs for Phase C readiness:
    - MultiIndex ['ZONEID','timestamp']
    - column presence per manifest
    - NA policy (features should have no NA; load target should have none; solar allows target NA only in clean/test)
    - per-zone hourly continuity (no gaps, no non-1h steps)
    - split ordering (train < val < test) per zone
    """
    issues: List[PrecheckIssue] = []
    t = manifest["tracks"][track]
    cols_policy = t["columns"]
    paths = t["paths"]

    split_paths = {}
    for split in ("clean", "train", "val", "test"):
        if split not in paths:
            issues.append(PrecheckIssue(track, split, "missing_path", f"manifest missing paths['{split}']"))
            continue
        split_paths[split] = _resolve_artifact_path(manifest_path, paths[split])
        if not split_paths[split].exists():
            issues.append(PrecheckIssue(track, split, "missing_file", f"file not found: {split_paths[split]}"))

    if any(i.kind == "missing_file" for i in issues):
        return False, issues, {}

    dfs: Dict[str, pd.DataFrame] = {}
    for split, p in split_paths.items():
        df = pd.read_parquet(p)
        dfs[split] = df
        if list(df.index.names) != ["ZONEID", "timestamp"]:
            issues.append(PrecheckIssue(track, split, "bad_index", f"index={df.index.names}"))

        for list_name in ("feature_cols", "target_cols", "hist_cols", "future_cols"):
            required_cols = cols_policy.get(list_name, [])
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                issues.append(
                    PrecheckIssue(track, split, "missing_columns", f"{list_name} missing: {missing[:10]}")
                )

        # Features should be fully observed (no NaNs).
        feature_cols = cols_policy.get("feature_cols", [])
        if feature_cols:
            na_features = _na_sum(df, feature_cols)
            if na_features != 0:
                issues.append(PrecheckIssue(track, split, "na_features", f"na_sum(feature_cols)={na_features}"))

        # Targets: load should have no missing; solar allows missing in clean/test (forecast month).
        target_cols = cols_policy.get("target_cols", [])
        na_targets = _na_sum(df, target_cols)
        if track == "load":
            if na_targets != 0:
                issues.append(PrecheckIssue(track, split, "na_targets", f"na_sum(target_cols)={na_targets}"))
        elif track == "solar":
            if split in ("train", "val") and na_targets != 0:
                issues.append(PrecheckIssue(track, split, "na_targets", f"na_sum(target_cols)={na_targets}"))
        else:
            pass

        # Hourly continuity checks per zone
        gap_total = 0
        non_hourly = 0
        for zone, g in df.groupby(level=0, sort=False):
            ts = pd.DatetimeIndex(g.droplevel(0).index)
            gap_total += _gap_hours_total(ts)
            non_hourly += _non_1h_steps(ts)
        if gap_total != 0:
            issues.append(PrecheckIssue(track, split, "gaps", f"gap_hours_total={gap_total}"))
        if non_hourly != 0:
            issues.append(PrecheckIssue(track, split, "non_1h_steps", f"non_1h_steps={non_hourly}"))

    # Split order checks per zone
    if all(k in dfs for k in ("train", "val", "test")):
        train_df, val_df, test_df = dfs["train"], dfs["val"], dfs["test"]
        for zone in train_df.index.get_level_values(0).unique():
            tr_max = train_df.xs(zone, level=0).index.max()
            va_min = val_df.xs(zone, level=0).index.min()
            te_min = test_df.xs(zone, level=0).index.min()
            if not (tr_max < va_min < te_min):
                issues.append(PrecheckIssue(track, "splits", "split_order", f"zone={zone} tr_max={tr_max} va_min={va_min} te_min={te_min}"))
            if (va_min - tr_max) != pd.Timedelta(hours=1):
                issues.append(PrecheckIssue(track, "splits", "boundary_gap", f"zone={zone} val_start-train_end={va_min - tr_max}"))

    # Optional: load weather columns should remain integer-like (quick interpolation smoke check).
    if check_fractional_weather and track == "load":
        clean_df = dfs.get("clean")
        if clean_df is not None:
            wcols = [c for c in clean_df.columns if c.startswith("w")]
            if wcols:
                arr = clean_df[wcols].to_numpy(dtype=np.float64, copy=False)
                # Detect any fractional part
                if np.any(np.mod(arr, 1.0) != 0.0):
                    issues.append(PrecheckIssue(track, "clean", "fractional_weather", "found non-integer w* values"))

    report = {
        "track": track,
        "n_issues": len(issues),
        "splits": {k: {"path": str(v)} for k, v in split_paths.items()},
        "columns": {k: list(cols_policy.get(k, [])) for k in ("feature_cols", "target_cols", "hist_cols", "future_cols")},
    }
    return len(issues) == 0, issues, report


def run_phase_c_precheck(
    manifest: str = "data/processed/dataset_manifest.json",
    tracks: Iterable[str] = ("load", "solar"),
    check_fractional_weather: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    mp = Path(manifest)
    m = _read_manifest(mp)
    summary: Dict[str, Any] = {"manifest": str(mp.resolve()), "tracks": {}, "ok": True}
    all_issues: List[Dict[str, str]] = []

    for track in tracks:
        if "tracks" not in m or track not in m["tracks"]:
            issue = PrecheckIssue(track, "manifest", "missing_track", "track missing in manifest['tracks']")
            all_issues.append(issue.__dict__)
            summary["ok"] = False
            continue

        ok, issues, report = check_track_splits(
            track=track, manifest_path=mp, manifest=m, check_fractional_weather=check_fractional_weather
        )
        summary["tracks"][track] = report
        if not ok:
            summary["ok"] = False
        for i in issues:
            all_issues.append(i.__dict__)

    summary["issues"] = all_issues
    return summary["ok"], summary


def main() -> None:
    p = argparse.ArgumentParser(description="Phase C precheck for Phase A2 artifacts (gaps/leakage/shape invariants).")
    p.add_argument("--manifest", type=str, default="data/processed/dataset_manifest.json")
    p.add_argument("--tracks", type=str, default="load,solar", help="Comma-separated list, e.g. load,solar")
    p.add_argument("--no_fractional_weather_check", action="store_true", help="Disable load w* integer-like check.")
    p.add_argument("--json_out", type=str, default="", help="Optional JSON path to write full report.")
    args = p.parse_args()

    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    ok, report = run_phase_c_precheck(
        manifest=args.manifest,
        tracks=tracks,
        check_fractional_weather=not args.no_fractional_weather_check,
    )

    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if ok:
        print("PHASE_C_PRECHECK_OK")
        return

    print("PHASE_C_PRECHECK_FAILED")
    for i in report.get("issues", []):
        print(f"- {i['track']}::{i['split']}::{i['kind']}: {i['detail']}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()


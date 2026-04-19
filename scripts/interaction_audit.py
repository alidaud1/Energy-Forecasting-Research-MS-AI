from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _topic_type(topic: str) -> str:
    if "/" not in topic:
        return topic
    return topic.split("/", 1)[0]


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _metric_delta(old: float, new: float) -> str:
    if not old:
        return "n/a"
    change = ((new - old) / old) * 100.0
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.2f}%"


def build_markdown(messages: List[Dict[str, Any]], *, run_id: str) -> str:
    rows = [m for m in messages if m.get("run_id") == run_id]

    # Core traces
    user_feedback = [m for m in rows if _topic_type(str(m.get("topic", ""))) == "user_feedback"]
    sys_directives = [m for m in rows if _topic_type(str(m.get("topic", ""))) == "system_directive"]
    directive_acks = [m for m in rows if _topic_type(str(m.get("topic", ""))) == "directive_ack"]
    directive_rejected = [m for m in rows if _topic_type(str(m.get("topic", ""))) == "directive_rejected"]
    human_gate = [m for m in rows if _topic_type(str(m.get("topic", ""))) == "human_gate_decision"]
    states = [m for m in rows if _topic_type(str(m.get("topic", ""))) == "state"]
    system_msgs = [m for m in rows if _topic_type(str(m.get("topic", ""))) == "system"]
    evals = [m for m in rows if _topic_type(str(m.get("topic", ""))) == "evaluate" and (m.get("payload") or {}).get("event") == "evaluation_complete"]

    # Proposed/final plan snapshots
    proposed_plan = {}
    for m in system_msgs:
        p = m.get("payload") or {}
        if p.get("event") == "run_started" and isinstance(p.get("plan"), dict):
            proposed_plan = dict(p.get("plan") or {})
            break

    finalized_plan = {}
    for m in states[::-1]:
        p = m.get("payload") or {}
        summ = (p.get("summary") if p.get("event") == "state_summary" else None) or {}
        plan = summ.get("plan")
        if isinstance(plan, dict):
            finalized_plan = dict(plan)
            break
    if not finalized_plan and proposed_plan:
        finalized_plan = dict(proposed_plan)

    # Metric table
    metric_rows: List[Tuple[str, str, str, str, Any, Any, str]] = []
    by_variant_track: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in evals:
        p = e.get("payload") or {}
        variant = str(p.get("variant") or "default")
        track = str(p.get("track") or "")
        fam = str(p.get("model_family") or "")
        name = str(p.get("model_name") or p.get("model_type") or "")
        m = p.get("metrics") or {}
        mae = m.get("mae")
        rmse = m.get("rmse")
        child = str(p.get("child_run_id") or "")
        metric_rows.append((variant, track, fam, name, mae, rmse, child))
        if isinstance(m, dict):
            by_variant_track[(variant, track)] = dict(m)

    lines: List[str] = []
    lines.append("# Interaction Audit")
    lines.append("")
    lines.append(f"Run ID: `{run_id}`")
    lines.append("")

    lines.append("## Human interventions")
    lines.append("")
    if not user_feedback:
        lines.append("No human interventions were recorded.")
    else:
        lines.append("| Stage | Raw human input |")
        lines.append("|---|---|")
        for m in user_feedback:
            p = m.get("payload") or {}
            stage = str(p.get("stage") or p.get("current_gate_stage") or "")
            text = str(p.get("human_text") or p.get("raw_text") or "")
            lines.append(f"| {stage} | {text} |")
    lines.append("")

    lines.append("## Human gate decisions")
    lines.append("")
    if not human_gate:
        lines.append("No human_gate_decision messages were recorded.")
    else:
        for m in human_gate:
            p = m.get("payload") or {}
            lines.append(f"- stage=`{p.get('stage')}` gate_id=`{p.get('gate_id')}` decision=`{p.get('decision')}` text=`{p.get('raw_text')}`")
    lines.append("")

    lines.append("## Structured directives")
    lines.append("")
    if not sys_directives:
        lines.append("No system_directive messages were generated.")
    else:
        for d in sys_directives:
            p = d.get("payload") or {}
            lines.append(f"### system_directive seq={d.get('seq')}")
            lines.append("")
            lines.append("```json")
            lines.append(_safe_json(p))
            lines.append("```")
            lines.append("")

    lines.append("## Applied overrides (acks)")
    lines.append("")
    if not directive_acks:
        lines.append("No directive_ack messages were recorded.")
    else:
        for a in directive_acks:
            p = a.get("payload") or {}
            lines.append("```json")
            lines.append(_safe_json(p))
            lines.append("```")
            lines.append("")

    lines.append("## Rejected overrides")
    lines.append("")
    if not directive_rejected:
        lines.append("No directive_rejected messages were recorded.")
    else:
        for a in directive_rejected:
            p = a.get("payload") or {}
            lines.append("```json")
            lines.append(_safe_json(p))
            lines.append("```")
            lines.append("")

    lines.append("## Proposed plan versus finalized plan")
    lines.append("")
    if proposed_plan:
        lines.append("### Proposed (run_started.plan)")
        lines.append("```json")
        lines.append(_safe_json(proposed_plan))
        lines.append("```")
        lines.append("")
    if finalized_plan:
        lines.append("### Finalized (latest state_summary.plan)")
        lines.append("```json")
        lines.append(_safe_json(finalized_plan))
        lines.append("```")
        lines.append("")

    lines.append("## Metric comparison")
    lines.append("")
    if not metric_rows:
        lines.append("No evaluation_complete messages found.")
    else:
        lines.append("| Variant | Track | Family | Model | MAE | RMSE | child_run_id |")
        lines.append("|---|---|---|---|---:|---:|---|")
        for v, t, fam, name, mae, rmse, child in sorted(metric_rows):
            lines.append(f"| {v} | {t} | {fam} | {name} | {mae} | {rmse} | {child} |")
        lines.append("")

        # Deltas for each track when both branches exist.
        tracks = sorted({t for (_v, t) in by_variant_track.keys() if t})
        any_delta = False
        for t in tracks:
            p0 = by_variant_track.get(("proposed", t))
            p1 = by_variant_track.get(("human_adjusted", t))
            if not (p0 and p1):
                continue
            any_delta = True
            try:
                mae0 = float(p0.get("mae"))
                mae1 = float(p1.get("mae"))
                rmse0 = float(p0.get("rmse"))
                rmse1 = float(p1.get("rmse"))
            except Exception:
                continue
            lines.append(f"### Impact of human guidance ({t})")
            lines.append("")
            lines.append(f"- MAE delta: `{_metric_delta(mae0, mae1)}`")
            lines.append(f"- RMSE delta: `{_metric_delta(rmse0, rmse1)}`")
            lines.append("")
        if not any_delta:
            lines.append("No proposed vs human_adjusted metric pair found in this run.")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Build thesis-style interaction_audit.md from logs/message_pool.jsonl")
    p.add_argument("--log", type=str, default="logs/message_pool.jsonl")
    p.add_argument("--run_id", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out)

    msgs = _load_jsonl(log_path)
    md = build_markdown(msgs, run_id=str(args.run_id))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote audit report: {out_path}")


if __name__ == "__main__":
    main()

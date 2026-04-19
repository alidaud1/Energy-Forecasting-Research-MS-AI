from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _topic_type(topic: str) -> str:
    if "/" not in topic:
        return topic
    return topic.split("/", 1)[0]


@dataclass(frozen=True)
class GateInfo:
    gate_id: str
    stage: str
    prompt: str
    correlation_id: str | None
    seq: int


def latest_active_gate(messages: List[Dict[str, Any]], *, run_id: str) -> Optional[GateInfo]:
    gates = [m for m in messages if m.get("run_id") == run_id and _topic_type(str(m.get("topic", ""))) == "gate"]
    # Find most recent gate_requested that doesn't have a later approved/rejected for same gate_id.
    reqs: List[Dict[str, Any]] = [m for m in gates if (m.get("payload") or {}).get("event") == "gate_requested"]
    if not reqs:
        return None
    reqs_sorted = sorted(reqs, key=lambda x: int(x.get("seq", -1)))
    for req in reversed(reqs_sorted):
        p = req.get("payload") or {}
        gid = str(p.get("gate_id") or "")
        stage = str(p.get("stage") or "")
        seq = int(req.get("seq", -1))
        decided = False
        for m in gates:
            pp = m.get("payload") or {}
            if str(pp.get("gate_id") or "") != gid:
                continue
            if (pp.get("event") in ("gate_approved", "gate_rejected")) and int(m.get("seq", -1)) > seq:
                decided = True
                break
        if not decided:
            return GateInfo(
                gate_id=gid,
                stage=stage,
                prompt=str(p.get("prompt") or ""),
                correlation_id=req.get("correlation_id"),
                seq=seq,
            )
    return None


def latest_artifacts(messages: List[Dict[str, Any]], *, run_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in messages:
        if m.get("run_id") != run_id:
            continue
        tt = _topic_type(str(m.get("topic", "")))
        if tt == "integrate" and (m.get("payload") or {}).get("event") in ("joint_report_ready", "final_research_summary_ready"):
            out["integration"] = dict(m.get("payload") or {})
    return out


def recent_messages(messages: List[Dict[str, Any]], *, run_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    rows = [m for m in messages if m.get("run_id") == run_id]
    rows.sort(key=lambda x: int(x.get("seq", -1)))
    out = rows[-limit:]
    # Keep it UI-friendly: drop huge payloads.
    trimmed = []
    for m in out:
        mm = dict(m)
        payload = mm.get("payload") or {}
        if isinstance(payload, dict) and len(str(payload)) > 4000:
            mm["payload"] = {"_truncated": True, "event": payload.get("event"), "keys": list(payload.keys())[:40]}
        trimmed.append(mm)
    return trimmed


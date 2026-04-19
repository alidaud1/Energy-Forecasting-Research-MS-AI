from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent, AgentConfig


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().split())


def _parse_human_command(text: str) -> Dict[str, Any]:
    """
    Minimal production-safe CLI parser:
    - approve / reject
    - set key=value [key=value ...]
    - enable baselines / disable baselines
    - run load only / run solar only / run both
    - help
    """
    raw = _normalize(text)
    low = raw.lower()

    if low in ("approve", "approved", "ok", "go", "yes", "y"):
        return {"action": "approve"}
    if low in ("reject", "no", "n", "stop", "cancel"):
        return {"action": "reject"}
    if low in ("help", "?"):
        return {"action": "help"}
    if low in ("enable baselines", "enable baseline", "baselines on"):
        return {"action": "set", "changes": {"with_baselines": True}}
    if low in ("disable baselines", "disable baseline", "baselines off"):
        return {"action": "set", "changes": {"with_baselines": False}}
    if low in ("run load only", "load only"):
        return {"action": "set", "changes": {"tracks": ["load"]}}
    if low in ("run solar only", "solar only"):
        return {"action": "set", "changes": {"tracks": ["solar"]}}
    if low in ("run both", "both"):
        return {"action": "set", "changes": {"tracks": ["load", "solar"]}}

    if low.startswith("set "):
        parts = raw[4:].strip().split()
        changes: Dict[str, Any] = {}
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            # basic typing
            if v.lower() in ("true", "false"):
                val: Any = v.lower() == "true"
            else:
                try:
                    if "." in v:
                        val = float(v)
                    else:
                        val = int(v)
                except Exception:
                    val = v
            changes[k] = val
        return {"action": "set", "changes": changes}

    return {"action": "unknown", "text": raw}

def _parse_postprocess_command(text: str) -> Dict[str, Any]:
    raw = _normalize(text)
    low = raw.lower()
    if low in ("accept", "approve", "ok", "go", "yes", "y"):
        return {"action": "accept"}
    if low in ("reject", "cancel", "stop", "no", "n"):
        return {"action": "reject"}

    parts = raw.split()
    if len(parts) < 3:
        return {"action": "unknown", "text": raw}

    action = parts[0].lower()
    target = parts[1].lower()
    if action not in ("scale", "bias") or target not in ("load", "solar"):
        return {"action": "unknown", "text": raw}

    kv: Dict[str, str] = {}
    for p in parts[2:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k.strip().lower()] = v.strip()

    out: Dict[str, Any] = {"action": action, "target": target}
    if "hours" in kv:
        out["hours"] = kv["hours"]
    if action == "scale":
        try:
            out["factor"] = float(kv.get("factor", ""))
        except Exception:
            return {"action": "unknown", "text": raw}
    else:
        try:
            out["add"] = float(kv.get("add", ""))
        except Exception:
            return {"action": "unknown", "text": raw}
    return out


@dataclass
class InteractiveResearchConfig:
    manifest_path: str = "data/processed/dataset_manifest.json"
    tracks: List[str] = field(default_factory=lambda: ["load", "solar"])

    # Phase D3: comparative audit grouping (proposed vs human-adjusted).
    comparison_group_id: str = ""

    # Windowing
    W: int = 168
    H: int = 24
    stride: int = 24

    # Baselines
    with_baselines: bool = True
    max_train_windows: int = 20000
    max_test_windows: int = 5000
    seed: int = 42

    # Transformers
    load_model: str = "patchtst"
    solar_model: str = "patchtst"
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 10


class InteractiveTaskManagerAgent(BaseAgent):
    """
    Phase D1 (HITL) task manager for research mode:
    - publishes prepare commands for required tracks
    - waits for PreparationAgent ready events
    - requests a single "model gate" approval from the human
    - after approval, publishes baseline + transformer train commands

    The orchestrator (CLI) is responsible for:
    - printing the gate prompt
    - collecting terminal input
    - publishing `user_feedback` events
    """

    GATE_ID_MODEL = "model_gate_v1"
    GATE_ID_PREP = "prep_gate_v1"
    GATE_ID_POSTPROCESS = "postprocess_gate_v1"

    def __init__(self, cfg: AgentConfig, pool, rcfg: InteractiveResearchConfig):
        super().__init__(cfg, pool)
        self.rcfg = rcfg

        self._started = False
        self._ready_payloads: Dict[str, dict] = {}
        self._gate_requested = False
        self._prep_approved = False
        self._approved = False
        self._rejected = False
        self._corr_id: str = ""
        self._initial_plan: Dict[str, Any] = {}
        self._plan_edited = False
        self._child_run_id: str = ""
        self._state_summary: Dict[str, Any] = {}
        self._directive_seen: set[str] = set()

        self.subscribe(topics=["system", "prepare", "user_feedback"], start_from="latest")

    def _plan_dict(self) -> Dict[str, Any]:
        if not self.rcfg.comparison_group_id:
            # Stable for the life of this interactive run.
            self.rcfg.comparison_group_id = f"cmp_{uuid.uuid4().hex[:8]}"
        return {
            "tracks": list(self.rcfg.tracks),
            "comparison_group_id": str(self.rcfg.comparison_group_id),
            "variant": "human_adjusted" if bool(self._plan_edited) else "proposed",
            "window": {"W": int(self.rcfg.W), "H": int(self.rcfg.H), "stride": int(self.rcfg.stride)},
            "with_baselines": bool(self.rcfg.with_baselines),
            "baseline_limits": {"max_train_windows": int(self.rcfg.max_train_windows), "max_test_windows": int(self.rcfg.max_test_windows), "seed": int(self.rcfg.seed)},
            "transformers": {
                "load_model": str(self.rcfg.load_model),
                "solar_model": str(self.rcfg.solar_model),
                "epochs": int(self.rcfg.epochs),
                "batch_size": int(self.rcfg.batch_size),
                "lr": float(self.rcfg.lr),
                "patience": int(self.rcfg.patience),
            },
        }

    def _utc_child_id(self, suffix: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"child_{ts}_{suffix}"

    def _diff_plans(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        # Simple shallow diff for audit prompts (keeps it one-screen).
        out = {}
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            if a.get(k) != b.get(k):
                out[k] = {"from": a.get(k), "to": b.get(k)}
        return out

    def _write_plan_diff_report(self, *, child_run_id: str) -> None:
        try:
            run_dir = Path("logs") / "runs" / self.run_id
            (run_dir / "reports").mkdir(parents=True, exist_ok=True)
            diff = self._diff_plans(self._initial_plan, self._plan_dict())
            p = run_dir / "reports" / "plan_edit_diff.md"
            lines = []
            lines.append(f"# Plan Edit Diff ({self.run_id} -> {child_run_id})\n\n")
            lines.append(f"- parent_run_id: `{self.run_id}`\n")
            lines.append(f"- child_run_id: `{child_run_id}`\n\n")
            if not diff:
                lines.append("No changes.\n")
            else:
                lines.append("Changed fields:\n")
                for k, v in diff.items():
                    lines.append(f"- {k}: `{v['from']}` -> `{v['to']}`\n")
            p.write_text("".join(lines), encoding="utf-8")
        except Exception:
            # Never fail the pipeline due to reporting.
            pass

    def _prep_gate_prompt(self) -> str:
        plan = self._plan_dict()
        lines: List[str] = []
        lines.append("PREP GATE (Phase D1)\n")
        lines.append("Confirm the prepared datasets are correct before training.\n\n")
        lines.append(f"- Manifest: {self.rcfg.manifest_path}\n")
        lines.append(f"- Tracks: {', '.join(plan['tracks'])}\n\n")

        for track in plan["tracks"]:
            ready = self._ready_payloads.get(track, {})
            summ = ready.get("summary", {}) or {}
            lines.append(f"{track.upper()} summary:\n")
            for split in ("train_scaled", "val_scaled", "test_scaled"):
                s = summ.get(split)
                if not s:
                    continue
                lines.append(
                    f"- {split}: rows={s.get('n_rows')} zones={s.get('n_zones')} "
                    f"ts=[{s.get('ts_min')} .. {s.get('ts_max')}] na_target={s.get('na_target')}\n"
                )
            lines.append("\n")

        lines.append("Commands:\n")
        lines.append("- approve | reject\n")
        lines.append("- run load only | run solar only | run both\n")
        lines.append("- enable baselines | disable baselines\n")
        lines.append("- set W=168 H=24 stride=24\n")
        return "".join(lines)

    def _gate_prompt(self) -> str:
        plan = self._plan_dict()
        lines = []
        lines.append("MODEL GATE (Phase D1)\n")
        lines.append("Approve or edit the training plan.\n\n")
        lines.append(f"- Tracks: {', '.join(plan['tracks'])}\n")
        w = plan["window"]
        lines.append(f"- Window: W={w['W']} H={w['H']} stride={w['stride']}\n")
        lines.append(f"- Baselines (XGBoost): {plan['with_baselines']}\n")
        t = plan["transformers"]
        lines.append(f"- Transformer load_model={t['load_model']} solar_model={t['solar_model']}\n")
        lines.append(f"- Train epochs={t['epochs']} batch_size={t['batch_size']} lr={t['lr']} patience={t['patience']}\n\n")
        lines.append("Commands:\n")
        lines.append("- approve | reject\n")
        lines.append("- set epochs=5 batch_size=128 lr=0.001 patience=2\n")
        lines.append("- set load_model=itransformer solar_model=patchtst\n")
        lines.append("- enable baselines | disable baselines\n")
        lines.append("- run load only | run solar only | run both\n")
        return "".join(lines)

    def _apply_changes(self, changes: Dict[str, Any]) -> None:
        # Apply changes first, then decide whether the plan is truly different from the initial plan.
        # Track selection
        if "tracks" in changes and isinstance(changes["tracks"], list):
            tr = [str(x) for x in changes["tracks"] if str(x) in ("load", "solar")]
            if tr:
                self.rcfg.tracks = tr

        # Baselines
        if "with_baselines" in changes:
            self.rcfg.with_baselines = bool(changes["with_baselines"])

        # Windowing
        for k in ("W", "H", "stride"):
            if k in changes:
                try:
                    setattr(self.rcfg, k, int(changes[k]))
                except Exception:
                    pass

        # Transformer plan
        for k in ("epochs", "batch_size", "patience"):
            if k in changes:
                try:
                    setattr(self.rcfg, k, int(changes[k]))
                except Exception:
                    pass
        if "lr" in changes:
            try:
                self.rcfg.lr = float(changes["lr"])
            except Exception:
                pass
        if "load_model" in changes:
            self.rcfg.load_model = str(changes["load_model"])
        if "solar_model" in changes:
            self.rcfg.solar_model = str(changes["solar_model"])

        # Baseline limits (optional)
        for k in ("max_train_windows", "max_test_windows", "seed"):
            if k in changes:
                try:
                    setattr(self.rcfg, k, int(changes[k]))
                except Exception:
                    pass

        # Mark as edited only if the plan meaningfully differs from what we started with.
        if self._initial_plan:
            self._plan_edited = bool(self._diff_plans(self._initial_plan, self._plan_dict()))

    def _publish_gate_request(self) -> None:
        self._gate_requested = True
        self.publish(
            topic_type="gate",
            kind="event",
            payload={
                "event": "gate_requested",
                "stage": "model",
                "gate_id": self.GATE_ID_MODEL,
                "prompt": self._gate_prompt(),
                "plan": self._plan_dict(),
            },
            correlation_id=self._corr_id,
        )

    def _publish_prep_gate_request(self) -> None:
        self.publish(
            topic_type="gate",
            kind="event",
            payload={
                "event": "gate_requested",
                "stage": "prep",
                "gate_id": self.GATE_ID_PREP,
                "prompt": self._prep_gate_prompt(),
                "plan": self._plan_dict(),
            },
            correlation_id=self._corr_id,
        )

    def _publish_prepare_for_tracks(self) -> None:
        for track in ("load", "solar"):
            if track not in self.rcfg.tracks:
                continue
            self.publish(
                topic_type="prepare",
                kind="command",
                payload={
                    "command": "prepare",
                    "manifest_path": self.rcfg.manifest_path,
                    "track": track,
                    "run_config": {
                        "track": track,
                        "model_type": "transformer",
                        "window": {"W": self.rcfg.W, "H": self.rcfg.H, "stride": self.rcfg.stride},
                        "training": {
                            "epochs": self.rcfg.epochs,
                            "batch_size": self.rcfg.batch_size,
                            "lr": self.rcfg.lr,
                            "patience": self.rcfg.patience,
                        },
                        "hpo": {"enabled": False, "n_trials": 1},
                        "interactive": True,
                    },
                },
                correlation_id=self._corr_id,
            )

    def _publish_training_commands(self) -> None:
        def publish_for_plan(*, plan: Dict[str, Any], child_run_id: str, parent_run_id: str, variant: str, corr_suffix: str) -> None:
            tracks = list(plan.get("tracks") or [])
            w = plan.get("window") or {}
            window = {"W": int(w.get("W", 168)), "H": int(w.get("H", 24)), "stride": int(w.get("stride", 24))}
            cmp_id = str(plan.get("comparison_group_id") or self.rcfg.comparison_group_id or "")

            base_limits = plan.get("baseline_limits") or {}
            base_training = {
                "max_train_windows": int(base_limits.get("max_train_windows", self.rcfg.max_train_windows)),
                "max_test_windows": int(base_limits.get("max_test_windows", self.rcfg.max_test_windows)),
                "seed": int(base_limits.get("seed", self.rcfg.seed)),
            }

            tfm = plan.get("transformers") or {}
            # Phase D2: allow limiting transformer work using the same window caps as baselines.
            ttraining = {
                "epochs": int(tfm.get("epochs", self.rcfg.epochs)),
                "batch_size": int(tfm.get("batch_size", self.rcfg.batch_size)),
                "lr": float(tfm.get("lr", self.rcfg.lr)),
                "patience": int(tfm.get("patience", self.rcfg.patience)),
                "max_train_windows": int(base_training["max_train_windows"]),
                "max_val_windows": int(base_training["max_test_windows"]),
                "max_test_windows": int(base_training["max_test_windows"]),
            }

            xgb_hp = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "random_state": int(base_training["seed"]),
                "tree_method": "hist",
            }

            # Baselines (direct to model developer)
            if bool(plan.get("with_baselines", True)):
                for track in ("load", "solar"):
                    if track not in tracks:
                        continue
                    ready = self._ready_payloads[track]
                    self.publish(
                        topic_type="train",
                        kind="command",
                        payload={
                            "command": "train",
                            "track": track,
                            "paths": ready["paths"],
                            "columns": ready["columns"],
                            "model_type": "xgboost",
                            "hyperparams": dict(xgb_hp),
                            "window": window,
                            "training": base_training,
                            "child_run_id": child_run_id,
                            "parent_run_id": parent_run_id,
                            "comparison_group_id": cmp_id,
                            "variant": variant,
                        },
                        correlation_id=f"{self._corr_id}_baseline_{track}_{corr_suffix}",
                        to_roles=["model_developer"],
                    )

            # Transformer plans (forecasting agents -> train_execute)
            if "load" in tracks:
                ready = self._ready_payloads["load"]
                self.publish(
                    topic_type="train",
                    kind="command",
                    payload={
                        "command": "train",
                        "track": "load",
                        "paths": ready["paths"],
                        "columns": ready["columns"],
                        "window": window,
                        "training": ttraining,
                        "model_name": str(tfm.get("load_model", self.rcfg.load_model)),
                        "strategy": "A",
                        "add_future_indicator": True,
                        "use_hour_embedding": False,
                        "child_run_id": child_run_id,
                        "parent_run_id": parent_run_id,
                        "comparison_group_id": cmp_id,
                        "variant": variant,
                    },
                    correlation_id=f"{self._corr_id}_transformer_load_{corr_suffix}",
                )

            if "solar" in tracks:
                ready = self._ready_payloads["solar"]
                self.publish(
                    topic_type="train",
                    kind="command",
                    payload={
                        "command": "train",
                        "track": "solar",
                        "paths": ready["paths"],
                        "columns": ready["columns"],
                        "window": window,
                        "training": ttraining,
                        "model_name": str(tfm.get("solar_model", self.rcfg.solar_model)),
                        "strategy": "B",
                        "add_future_indicator": True,
                        "use_hour_embedding": True,
                        "child_run_id": child_run_id,
                        "parent_run_id": parent_run_id,
                        "comparison_group_id": cmp_id,
                        "variant": variant,
                    },
                    correlation_id=f"{self._corr_id}_transformer_solar_{corr_suffix}",
                )

        # Phase D3: if the human edited the plan, run both branches under the same comparison_group_id:
        # 1) proposed = original plan snapshot
        # 2) human_adjusted = current edited plan (child run)
        if self._plan_edited and self._initial_plan:
            proposed_child = self._utc_child_id("proposed")
            adjusted_child = self._child_run_id or self._utc_child_id("planedit")
            self._child_run_id = adjusted_child
            # Proposed branch
            publish_for_plan(plan=dict(self._initial_plan), child_run_id=proposed_child, parent_run_id=self.run_id, variant="proposed", corr_suffix="proposed")
            # Adjusted branch
            publish_for_plan(plan=self._plan_dict(), child_run_id=adjusted_child, parent_run_id=self.run_id, variant="human_adjusted", corr_suffix="adjusted")
            return

        # Default: single proposed branch
        publish_for_plan(plan=self._plan_dict(), child_run_id=self._child_run_id or "", parent_run_id=self.run_id if self._child_run_id else "", variant="proposed", corr_suffix="single")

    def _publish_human_gate_decision(self, *, stage: str, decision: str, raw_text: str, gate_id: str) -> None:
        """
        Phase D3 compatibility signal: publish a normalized decision event the orchestrator/audit can use.
        """
        self.publish(
            topic_type="human_gate_decision",
            kind="event",
            payload={
                "event": "human_gate_decision",
                "stage": stage,
                "gate_id": gate_id,
                "decision": decision,
                "raw_text": (raw_text or "")[:500],
                "comparison_group_id": str(self.rcfg.comparison_group_id or ""),
            },
            correlation_id=self._corr_id,
        )

    def _publish_system_directive(self, *, stage: str, gate_id: str, raw_text: str, overrides: Dict[str, Any], reason: str) -> None:
        """
        Phase D3 compatibility signal: publish a structured directive message.
        (Our pipeline still uses `directive/*` for postprocess; this is additive for audit + future wiring.)
        """
        if not overrides:
            return
        directive_key = json.dumps({"stage": stage, "gate_id": gate_id, "overrides": overrides}, sort_keys=True, default=str)
        if directive_key in self._directive_seen:
            return
        self._directive_seen.add(directive_key)
        self.publish(
            topic_type="system_directive",
            kind="event",
            payload={
                "event": "system_directive",
                "stage": stage,
                "gate_id": gate_id,
                "target_roles": ["*"],
                "overrides": overrides,
                "reason": reason,
                "source_text": (raw_text or "")[:500],
                "comparison_group_id": str(self.rcfg.comparison_group_id or ""),
                "variant": "human_adjusted" if bool(self._plan_edited) else "proposed",
            },
            correlation_id=self._corr_id,
        )

    def tick(self) -> None:
        # Start once.
        if not self._started:
            self._started = True
            self._corr_id = str(uuid.uuid4())
            self._initial_plan = self._plan_dict()
            self.publish(
                topic_type="system",
                kind="event",
                payload={
                    "event": "run_started",
                    "mode": "research_interactive",
                    "plan": self._plan_dict(),
                },
                correlation_id=self._corr_id,
            )
            self._publish_prepare_for_tracks()
            self._publish_state_summary(event="state_initialized")

        # Process incoming messages (ready + user feedback)
        for msg in self.poll(max_messages=200):
            # Ready events
            if msg.topic.startswith(f"prepare/{self.run_id}") and msg.kind == "event" and msg.payload.get("event") == "ready":
                track = msg.payload.get("track")
                if track in ("load", "solar") and track in self.rcfg.tracks:
                    self._ready_payloads[track] = msg.payload

            # Human feedback
            if msg.topic.startswith(f"user_feedback/{self.run_id}") and msg.kind == "event":
                payload = msg.payload or {}
                gate_id = payload.get("gate_id")
                if gate_id not in (self.GATE_ID_PREP, self.GATE_ID_MODEL, self.GATE_ID_POSTPROCESS):
                    continue

                # Translate human text into structured directives here (TaskManager is the boundary).
                if gate_id == self.GATE_ID_POSTPROCESS:
                    cmd = _parse_postprocess_command(payload.get("human_text", ""))
                    self._publish_human_gate_decision(
                        stage="postprocess",
                        gate_id=gate_id,
                        decision="change_requested" if cmd.get("action") not in ("accept", "reject") else cmd.get("action"),
                        raw_text=str(payload.get("human_text", "")),
                    )
                    self._publish_state_summary(event="directive_parsed", gate_id=gate_id, stage="postprocess", human_text=payload.get("human_text", ""), structured_command=cmd)
                    self.publish(
                        topic_type="directive",
                        kind="event",
                        payload={
                            "event": "directive",
                            "gate_id": gate_id,
                            "stage": "postprocess",
                            "human_text": payload.get("human_text", ""),
                            "structured_command": cmd,
                        },
                        correlation_id=self._corr_id,
                        to_roles=["forecast_integration"],
                    )
                    # Phase D3 compatibility: also emit a generic directive with dotted-ish keys.
                    overrides: Dict[str, Any] = {}
                    if cmd.get("action") == "scale":
                        overrides["postprocess.action"] = "scale"
                        overrides["postprocess.target"] = cmd.get("target")
                        overrides["postprocess.factor"] = cmd.get("factor")
                        if cmd.get("hours"):
                            overrides["postprocess.hours"] = cmd.get("hours")
                    if cmd.get("action") == "bias":
                        overrides["postprocess.action"] = "bias"
                        overrides["postprocess.target"] = cmd.get("target")
                        overrides["postprocess.add"] = cmd.get("add")
                        if cmd.get("hours"):
                            overrides["postprocess.hours"] = cmd.get("hours")
                    self._publish_system_directive(
                        stage="postprocess",
                        gate_id=gate_id,
                        raw_text=str(payload.get("human_text", "")),
                        overrides=overrides,
                        reason="Human postprocess command.",
                    )
                    continue

                cmd = _parse_human_command(payload.get("human_text", ""))
                self._publish_state_summary(event="directive_parsed", gate_id=gate_id, stage="prep" if gate_id == self.GATE_ID_PREP else "model", human_text=payload.get("human_text", ""), structured_command=cmd)
                if cmd.get("action") == "set":
                    self._publish_human_gate_decision(
                        stage="prep" if gate_id == self.GATE_ID_PREP else "model",
                        gate_id=gate_id,
                        decision="change_requested",
                        raw_text=str(payload.get("human_text", "")),
                    )
                    # Best-effort mapping for audit/future use.
                    overrides = {f"plan.{k}": v for k, v in (cmd.get("changes", {}) or {}).items()}
                    self._publish_system_directive(
                        stage="prep" if gate_id == self.GATE_ID_PREP else "model",
                        gate_id=gate_id,
                        raw_text=str(payload.get("human_text", "")),
                        overrides=overrides,
                        reason="Human requested plan override via CLI.",
                    )
                self.publish(
                    topic_type="directive",
                    kind="event",
                    payload={
                        "event": "directive",
                        "gate_id": gate_id,
                        "stage": "prep" if gate_id == self.GATE_ID_PREP else "model",
                        "human_text": payload.get("human_text", ""),
                        "structured_command": cmd,
                    },
                    correlation_id=self._corr_id,
                )
                if cmd["action"] == "help":
                    if gate_id == self.GATE_ID_PREP:
                        self._publish_prep_gate_request()
                    else:
                        self._publish_gate_request()
                    continue
                if cmd["action"] == "set":
                    self._apply_changes(cmd.get("changes", {}) or {})
                    # Re-run prepare if track selection changed and we lack ready payloads.
                    for t in list(self.rcfg.tracks):
                        if t not in self._ready_payloads:
                            self._publish_prepare_for_tracks()
                            break
                    if gate_id == self.GATE_ID_PREP:
                        self._publish_prep_gate_request()
                    else:
                        self._publish_gate_request()
                    continue
                if cmd["action"] == "approve":
                    self._publish_human_gate_decision(
                        stage="prep" if gate_id == self.GATE_ID_PREP else "model",
                        gate_id=gate_id,
                        decision="approve",
                        raw_text=str(payload.get("human_text", "")),
                    )
                    if gate_id == self.GATE_ID_PREP:
                        # Only approve when all selected tracks are ready.
                        if not all(t in self._ready_payloads for t in self.rcfg.tracks):
                            self._publish_prep_gate_request()
                            continue
                        self._prep_approved = True
                        self._publish_state_summary(event="gate_approved", gate_id=gate_id, stage="prep")
                        self.publish(
                            topic_type="gate",
                            kind="event",
                            payload={"event": "gate_approved", "stage": "prep", "gate_id": self.GATE_ID_PREP, "plan": self._plan_dict()},
                            correlation_id=self._corr_id,
                        )
                    else:
                        self._approved = True
                elif cmd["action"] == "reject":
                    self._publish_human_gate_decision(
                        stage="prep" if gate_id == self.GATE_ID_PREP else "model",
                        gate_id=gate_id,
                        decision="reject",
                        raw_text=str(payload.get("human_text", "")),
                    )
                    self._rejected = True
                    self._publish_state_summary(event="gate_rejected", gate_id=gate_id, stage="prep" if gate_id == self.GATE_ID_PREP else "model")
                else:
                    # Unknown: re-prompt with the same plan.
                    if gate_id == self.GATE_ID_PREP:
                        self._publish_prep_gate_request()
                    else:
                        self._publish_gate_request()

        # Prep gate request once all required tracks are ready.
        if not self._prep_approved and not self._rejected:
            if all(t in self._ready_payloads for t in self.rcfg.tracks):
                # Only request once per plan state.
                if not getattr(self, "_prep_gate_requested", False):
                    setattr(self, "_prep_gate_requested", True)
                    self._publish_prep_gate_request()

        # Model gate request only after prep approval.
        if self._prep_approved and not self._gate_requested and not self._approved and not self._rejected:
            if all(t in self._ready_payloads for t in self.rcfg.tracks):
                self._publish_gate_request()

        # If approved and not yet launched, publish training commands.
        if self._approved and not getattr(self, "_launched", False):
            setattr(self, "_launched", True)
            if self._plan_edited:
                self._child_run_id = self._utc_child_id("planedit")
                self._write_plan_diff_report(child_run_id=self._child_run_id)
                self._publish_state_summary(event="child_run_created", child_run_id=self._child_run_id, parent_run_id=self.run_id)
                self.publish(
                    topic_type="system",
                    kind="event",
                    payload={
                        "event": "child_run_created",
                        "parent_run_id": self.run_id,
                        "child_run_id": self._child_run_id,
                        "plan": self._plan_dict(),
                    },
                    correlation_id=self._corr_id,
                )
            self.publish(
                topic_type="gate",
                kind="event",
                payload={"event": "gate_approved", "stage": "model", "gate_id": self.GATE_ID_MODEL, "plan": self._plan_dict()},
                correlation_id=self._corr_id,
            )
            self._publish_state_summary(event="gate_approved", gate_id=self.GATE_ID_MODEL, stage="model")
            self._publish_training_commands()

        if self._rejected and not getattr(self, "_stopped", False):
            setattr(self, "_stopped", True)
            self.publish(
                topic_type="gate",
                kind="event",
                payload={"event": "gate_rejected", "stage": "model", "gate_id": self.GATE_ID_MODEL, "plan": self._plan_dict()},
                correlation_id=self._corr_id,
                level="warning",
            )
            # Orchestrator will stop when it sees this error-like condition (interactive cancel).

    def _publish_state_summary(
        self,
        *,
        event: str,
        gate_id: str = "",
        stage: str = "",
        human_text: str = "",
        structured_command: Optional[Dict[str, Any]] = None,
        child_run_id: str = "",
        parent_run_id: str = "",
    ) -> None:
        """
        Phase D2: memory compression.
        Publish a compact structured state summary after each interaction event so later stages
        don't need the full chat history.
        """
        summary = {
            "event": event,
            "approved": {"prep": bool(self._prep_approved), "model": bool(self._approved), "postprocess": False},
            "plan": self._plan_dict(),
            "gate_id": gate_id,
            "stage": stage,
            "last_human_text": (human_text or "")[:200],
            "last_structured_command": structured_command or {},
            "plan_edited": bool(self._plan_edited),
            "child_run_id": child_run_id or self._child_run_id,
            "parent_run_id": parent_run_id or (self.run_id if (child_run_id or self._child_run_id) else ""),
        }
        self._state_summary = summary
        self.publish(topic_type="state", kind="event", payload={"event": "state_summary", "summary": summary}, correlation_id=self._corr_id)

from __future__ import annotations

import argparse
from datetime import datetime
import time
import sys

from src.agents.base_agent import AgentConfig
from src.agents.message_pool import MessagePool
from src.agents.model_developer_agent import ModelDeveloperAgent
from src.agents.model_manager_agent import ModelManagerAgent
from src.agents.preparation_agent import PreparationAgent
from src.agents.task_manager_agent import RunConfig, TaskManagerAgent
from src.agents.load_forecasting_agent import LoadForecastingAgent
from src.agents.solar_forecasting_agent import SolarForecastingAgent
from src.agents.forecast_integration_agent import ForecastIntegrationAgent
from src.agents.messages import Message, TopicId
from src.phase_c_precheck import run_phase_c_precheck


def make_run_id() -> str:
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


def parse_args():
    p = argparse.ArgumentParser(description="Orchestrator: baseline (Phase B2) or research (Phase C2)")
    p.add_argument("--mode", type=str, choices=["baseline", "research"], default="baseline")
    p.add_argument("--manifest", type=str, default="data/processed/dataset_manifest.json")
    p.add_argument("--track", type=str, choices=["load", "solar"], default="load", help="Used in baseline mode.")
    p.add_argument("--model", type=str, choices=["linear", "xgboost"], default="xgboost", help="Used in baseline mode.")

    p.add_argument("--W", type=int, default=168)
    p.add_argument("--H", type=int, default=24)
    p.add_argument("--stride", type=int, default=24)

    p.add_argument("--max_train_windows", type=int, default=20000)
    p.add_argument("--max_test_windows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hpo", action="store_true", help="Enable Optuna skeleton (1 trial by default)")
    p.add_argument("--n_trials", type=int, default=1)

    # Research mode (transformers)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--load_model", type=str, choices=["patchtst", "itransformer"], default="patchtst")
    p.add_argument("--solar_model", type=str, choices=["patchtst", "itransformer"], default="patchtst")
    p.add_argument("--with_baselines", action="store_true", help="In research mode, also run XGBoost baselines for both tracks.")

    p.add_argument("--tick_seconds", type=float, default=1.0)
    p.add_argument("--timeout_seconds", type=float, default=600.0)
    p.add_argument("--print_messages", action="store_true", help="Print live message flow to console.")
    p.add_argument("--print_every", type=float, default=5.0, help="Seconds between status prints.")
    p.add_argument(
        "--no_precheck",
        action="store_true",
        help="Disable Phase C precheck (artifact invariants) before starting the run.",
    )
    return p.parse_args()


def main():
    # Reduce "no output" confusion on Windows shells.
    try:  # pragma: no cover
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args()
    run_id = make_run_id()

    pool = MessagePool(log_path="logs/message_pool.jsonl", load_existing=True)

    agents = []

    if args.mode == "baseline":
        if not args.no_precheck:
            ok, report = run_phase_c_precheck(manifest=args.manifest, tracks=(args.track,), check_fractional_weather=True)
            if not ok:
                print("[ERROR] Phase C precheck failed (baseline mode). Fix Phase A artifacts first.", flush=True)
                for i in report.get("issues", []):
                    print(f"- {i['track']}::{i['split']}::{i['kind']}: {i['detail']}", flush=True)
                raise SystemExit(2)

        run_cfg = RunConfig(
            manifest_path=args.manifest,
            track=args.track,
            model_type=args.model,
            window={"W": args.W, "H": args.H, "stride": args.stride},
            training={
                "max_train_windows": args.max_train_windows,
                "max_test_windows": args.max_test_windows,
                "seed": args.seed,
            },
            hpo={
                "enabled": bool(args.hpo),
                "n_trials": int(args.n_trials),
                "study_name": f"study_{run_id}",
                "storage": None,
            },
        )

        task_manager = TaskManagerAgent(
            AgentConfig(agent_id="TaskManagerAgent", role="task_manager", run_id=run_id),
            pool,
            run_cfg,
        )
        prep = PreparationAgent(AgentConfig(agent_id="PreparationAgent", role="preparation", run_id=run_id), pool)
        model_manager = ModelManagerAgent(AgentConfig(agent_id="ModelManagerAgent", role="model_manager", run_id=run_id), pool)
        model_dev = ModelDeveloperAgent(AgentConfig(agent_id="ModelDeveloperAgent", role="model_developer", run_id=run_id), pool)
        agents = [task_manager, prep, model_manager, model_dev]

    else:
        if not args.no_precheck:
            ok, report = run_phase_c_precheck(manifest=args.manifest, tracks=("load", "solar"), check_fractional_weather=True)
            if not ok:
                print("[ERROR] Phase C precheck failed (research mode). Fix Phase A artifacts first.", flush=True)
                for i in report.get("issues", []):
                    print(f"- {i['track']}::{i['split']}::{i['kind']}: {i['detail']}", flush=True)
                raise SystemExit(2)

        common_window = {"W": args.W, "H": args.H, "stride": args.stride}
        common_training = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "patience": args.patience}

        tm_load = TaskManagerAgent(
            AgentConfig(agent_id="TaskManagerLoad", role="task_manager", run_id=run_id),
            pool,
            RunConfig(
                manifest_path=args.manifest,
                track="load",
                model_type="transformer",
                window=common_window,
                training=common_training,
                hpo={"enabled": False, "n_trials": 1},
            ),
        )
        tm_solar = TaskManagerAgent(
            AgentConfig(agent_id="TaskManagerSolar", role="task_manager", run_id=run_id),
            pool,
            RunConfig(
                manifest_path=args.manifest,
                track="solar",
                model_type="transformer",
                window=common_window,
                training=common_training,
                hpo={"enabled": False, "n_trials": 1},
            ),
        )
        prep = PreparationAgent(AgentConfig(agent_id="PreparationAgent", role="preparation", run_id=run_id), pool)
        load_agent = LoadForecastingAgent(AgentConfig(agent_id="LoadForecastingAgent", role="load_forecasting", run_id=run_id), pool)
        solar_agent = SolarForecastingAgent(AgentConfig(agent_id="SolarForecastingAgent", role="solar_forecasting", run_id=run_id), pool)
        dev = ModelDeveloperAgent(AgentConfig(agent_id="ModelDeveloperAgent", role="model_developer", run_id=run_id), pool)
        integ = ForecastIntegrationAgent(AgentConfig(agent_id="ForecastIntegrationAgent", role="forecast_integration", run_id=run_id), pool)

        agents = [tm_load, tm_solar, prep, load_agent, solar_agent, dev, integ]

    start = time.time()
    done = False
    failed = False
    # If the pool already contains older runs, avoid dumping historical messages.
    last_seq_printed = pool.get_all()[-1].seq if (args.print_messages and pool.size() > 0) else -1
    last_status_print = 0.0

    if args.mode == "baseline":
        print(f"[START] mode=baseline run_id={run_id} track={args.track} model={args.model}", flush=True)
    else:
        print(f"[START] mode=research run_id={run_id} load_model={args.load_model} solar_model={args.solar_model}", flush=True)

    ready = {"load": False, "solar": False}
    injected_train = False

    while not done and (time.time() - start) < args.timeout_seconds:
        for a in agents:
            a.tick()

        # Live message printing (helps see progress in long training)
        if args.print_messages:
            msgs = pool.get_all()
            new = [m for m in msgs if m.seq > last_seq_printed]
            for m in new[-200:]:
                print(
                    f"[MSG] seq={m.seq} topic={m.topic} kind={m.kind} sender={m.sender} level={m.level} corr={m.correlation_id}",
                    flush=True,
                )
            if new:
                last_seq_printed = max(m.seq for m in new)

        now = time.time()
        if (now - start) - last_status_print >= args.print_every:
            last_status_print = now - start
            print(f"[STATUS] elapsed={now-start:.1f}s pool_size={pool.size()} run_id={run_id}", flush=True)

        # Stop early on any error message in prepare/train/evaluate for this run
        for topic_type in ("prepare", "train", "train_execute", "evaluate"):
            for m in pool.find_by_topic(topic_type, topic_source=run_id):
                if m.kind == "error":
                    print(f"[ERROR] topic={m.topic} sender={m.sender} payload={m.payload}", flush=True)
                    failed = True
                    done = True
                    break
            if done:
                break

        if args.mode == "research":
            prep_msgs = pool.find_by_topic("prepare", topic_source=run_id)
            for m in prep_msgs:
                if m.kind == "event" and m.payload.get("event") == "ready":
                    tr = m.payload.get("track")
                    if tr in ready:
                        ready[tr] = True

            if ready["load"] and ready["solar"] and not injected_train:
                injected_train = True
                load_ready_msg = next(
                    m for m in prep_msgs if m.kind == "event" and m.payload.get("event") == "ready" and m.payload.get("track") == "load"
                )
                solar_ready_msg = next(
                    m for m in prep_msgs if m.kind == "event" and m.payload.get("event") == "ready" and m.payload.get("track") == "solar"
                )
                corr_id = f"{run_id}_train"

                # Optional: run baselines in the same run_id (so integration can compare error distributions).
                if args.with_baselines:
                    xgb_hp = {
                        "n_estimators": 300,
                        "max_depth": 6,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_lambda": 1.0,
                        "random_state": args.seed,
                        "tree_method": "hist",
                    }
                    base_training = {
                        "max_train_windows": args.max_train_windows,
                        "max_test_windows": args.max_test_windows,
                        "seed": args.seed,
                    }
                    for trk, ready_msg in (("load", load_ready_msg), ("solar", solar_ready_msg)):
                        pool.publish(
                            Message.new(
                                topic=TopicId("train", run_id),
                                kind="command",
                                sender="Orchestrator",
                                sender_role="system",
                                to_roles=["model_developer"],
                                payload={
                                    "command": "train",
                                    "track": trk,
                                    "paths": ready_msg.payload["paths"],
                                    "columns": ready_msg.payload["columns"],
                                    "model_type": "xgboost",
                                    "hyperparams": dict(xgb_hp),
                                    "window": common_window,
                                    "training": base_training,
                                },
                                correlation_id=f"{corr_id}_baseline_{trk}",
                                run_id=run_id,
                            )
                        )

                pool.publish(
                    Message.new(
                        topic=TopicId("train", run_id),
                        kind="command",
                        sender="Orchestrator",
                        sender_role="system",
                        payload={
                            "command": "train",
                            "track": "load",
                            "paths": load_ready_msg.payload["paths"],
                            "columns": load_ready_msg.payload["columns"],
                            "window": common_window,
                            "training": common_training,
                            "model_name": args.load_model,
                            "strategy": "A",
                            "add_future_indicator": True,
                            "use_hour_embedding": False,
                        },
                        correlation_id=corr_id,
                        run_id=run_id,
                    )
                )
                pool.publish(
                    Message.new(
                        topic=TopicId("train", run_id),
                        kind="command",
                        sender="Orchestrator",
                        sender_role="system",
                        payload={
                            "command": "train",
                            "track": "solar",
                            "paths": solar_ready_msg.payload["paths"],
                            "columns": solar_ready_msg.payload["columns"],
                            "window": common_window,
                            "training": common_training,
                            "model_name": args.solar_model,
                            "strategy": "B",
                            "add_future_indicator": True,
                            "use_hour_embedding": True,
                        },
                        correlation_id=corr_id,
                        run_id=run_id,
                    )
                )

        eval_msgs = pool.find_by_topic("evaluate", topic_source=run_id)
        for m in eval_msgs:
            if m.kind == "result" and m.payload.get("event") == "evaluation_complete":
                if args.mode == "baseline":
                    done = True
                    break

        if args.mode == "research":
            integ_msgs = pool.find_by_topic("integrate", topic_source=run_id)
            if any(
                m.kind == "result" and m.payload.get("event") in ("joint_report_ready", "final_research_summary_ready")
                for m in integ_msgs
            ):
                done = True

        time.sleep(args.tick_seconds)

    if not done:
        print(f"[WARN] Timeout reached without completion. run_id={run_id}", flush=True)
        return

    if failed:
        print(f"[FAIL] Run failed. run_id={run_id}", flush=True)
        raise SystemExit(2)

    if args.mode == "baseline":
        print(f"[OK] Run completed. run_id={run_id}", flush=True)
        print("See logs/message_pool.jsonl and logs/runs/<run_id>/ for artifacts.", flush=True)
    else:
        print(f"[OK] Research run completed. run_id={run_id}", flush=True)
        print(f"Check logs/runs/{run_id}/reports/joint_decision_report.md", flush=True)
        print("Message trace: logs/message_pool.jsonl", flush=True)


if __name__ == "__main__":
    main()

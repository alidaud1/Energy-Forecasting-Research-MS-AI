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


def make_run_id() -> str:
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


def parse_args():
    p = argparse.ArgumentParser(description="Phase B Orchestrator: agentic baseline training loop")
    p.add_argument("--manifest", type=str, default="data/processed/dataset_manifest.json")
    p.add_argument("--track", type=str, choices=["load", "solar"], default="load")
    p.add_argument("--model", type=str, choices=["linear", "xgboost"], default="xgboost")

    p.add_argument("--W", type=int, default=168)
    p.add_argument("--H", type=int, default=24)
    p.add_argument("--stride", type=int, default=24)

    p.add_argument("--max_train_windows", type=int, default=20000)
    p.add_argument("--max_test_windows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hpo", action="store_true", help="Enable Optuna skeleton (1 trial by default)")
    p.add_argument("--n_trials", type=int, default=1)

    p.add_argument("--tick_seconds", type=float, default=1.0)
    p.add_argument("--timeout_seconds", type=float, default=600.0)
    p.add_argument("--print_messages", action="store_true", help="Print live message flow to console.")
    p.add_argument("--print_every", type=float, default=5.0, help="Seconds between status prints.")
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
    prep = PreparationAgent(
        AgentConfig(agent_id="PreparationAgent", role="preparation", run_id=run_id),
        pool,
    )
    model_manager = ModelManagerAgent(
        AgentConfig(agent_id="ModelManagerAgent", role="model_manager", run_id=run_id),
        pool,
    )
    model_dev = ModelDeveloperAgent(
        AgentConfig(agent_id="ModelDeveloperAgent", role="model_developer", run_id=run_id),
        pool,
    )

    agents = [task_manager, prep, model_manager, model_dev]

    start = time.time()
    done = False
    last_seq_printed = -1
    last_status_print = 0.0

    print(f"[START] run_id={run_id} track={args.track} model={args.model}", flush=True)

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
        for topic_type in ("prepare", "train", "evaluate"):
            for m in pool.find_by_topic(topic_type, topic_source=run_id):
                if m.kind == "error":
                    print(f"[ERROR] topic={m.topic} sender={m.sender} payload={m.payload}", flush=True)
                    done = True
                    break
            if done:
                break

        eval_msgs = pool.find_by_topic("evaluate", topic_source=run_id)
        for m in eval_msgs:
            if m.kind == "result" and m.payload.get("event") == "evaluation_complete":
                done = True
                break

        time.sleep(args.tick_seconds)

    if not done:
        print(f"[WARN] Timeout reached without completion. run_id={run_id}", flush=True)
        return

    print(f"[OK] Run completed. run_id={run_id}", flush=True)
    print("See logs/message_pool.jsonl and logs/runs/<run_id>/ for artifacts.", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import sys

from src.agents.base_agent import AgentConfig
from src.agents.interactive_task_manager_agent import InteractiveResearchConfig, InteractiveTaskManagerAgent
from src.agents.load_forecasting_agent import LoadForecastingAgent
from src.agents.message_pool import MessagePool
from src.agents.model_developer_agent import ModelDeveloperAgent
from src.agents.preparation_agent import PreparationAgent
from src.agents.solar_forecasting_agent import SolarForecastingAgent
from src.agents.forecast_integration_agent import ForecastIntegrationAgent
from src.agents.messages import Message, TopicId
from src.phase_c_precheck import run_phase_c_precheck


def make_run_id() -> str:
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


def parse_args():
    p = argparse.ArgumentParser(description="Async Orchestrator (Phase D2): interactive research runs with async input.")
    p.add_argument("--manifest", type=str, default="data/processed/dataset_manifest.json")
    p.add_argument("--W", type=int, default=168)
    p.add_argument("--H", type=int, default=24)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--load_model", type=str, choices=["patchtst", "itransformer"], default="patchtst")
    p.add_argument("--solar_model", type=str, choices=["patchtst", "itransformer"], default="patchtst")
    p.add_argument("--with_baselines", action="store_true")
    p.add_argument("--max_train_windows", type=int, default=20000)
    p.add_argument("--max_test_windows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tick_seconds", type=float, default=1.0)
    p.add_argument("--timeout_seconds", type=float, default=7200.0)
    p.add_argument("--print_messages", action="store_true")
    p.add_argument("--no_precheck", action="store_true")
    return p.parse_args()


async def _prompt_line(prompt: str) -> str:
    """
    Prefer prompt_toolkit for clean async prompts; fall back to a thread.
    """
    try:
        from prompt_toolkit import PromptSession  # type: ignore
        from prompt_toolkit.patch_stdout import patch_stdout  # type: ignore

        session = PromptSession()
        with patch_stdout():
            return (await session.prompt_async(prompt)).strip()
    except Exception:
        # Fallback: run blocking stdin in a worker thread.
        return (await asyncio.to_thread(lambda: sys.stdin.readline())).strip()


async def main_async() -> None:
    try:  # pragma: no cover
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args()
    run_id = make_run_id()

    pool = MessagePool(log_path="logs/message_pool.jsonl", load_existing=True)

    if not args.no_precheck:
        ok, report = run_phase_c_precheck(manifest=args.manifest, tracks=("load", "solar"), check_fractional_weather=True)
        if not ok:
            print("[ERROR] Phase C precheck failed. Fix Phase A artifacts first.", flush=True)
            for i in report.get("issues", []):
                print(f"- {i['track']}::{i['split']}::{i['kind']}: {i['detail']}", flush=True)
            raise SystemExit(2)

    tm = InteractiveTaskManagerAgent(
        AgentConfig(agent_id="InteractiveTaskManagerAgent", role="task_manager", run_id=run_id),
        pool,
        InteractiveResearchConfig(
            manifest_path=args.manifest,
            tracks=["load", "solar"],
            W=args.W,
            H=args.H,
            stride=args.stride,
            with_baselines=bool(args.with_baselines),
            max_train_windows=int(args.max_train_windows),
            max_test_windows=int(args.max_test_windows),
            seed=int(args.seed),
            load_model=str(args.load_model),
            solar_model=str(args.solar_model),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            patience=int(args.patience),
        ),
    )
    prep = PreparationAgent(AgentConfig(agent_id="PreparationAgent", role="preparation", run_id=run_id), pool)
    load_agent = LoadForecastingAgent(AgentConfig(agent_id="LoadForecastingAgent", role="load_forecasting", run_id=run_id), pool)
    solar_agent = SolarForecastingAgent(AgentConfig(agent_id="SolarForecastingAgent", role="solar_forecasting", run_id=run_id), pool)
    dev = ModelDeveloperAgent(AgentConfig(agent_id="ModelDeveloperAgent", role="model_developer", run_id=run_id), pool)
    integ = ForecastIntegrationAgent(AgentConfig(agent_id="ForecastIntegrationAgent", role="forecast_integration", run_id=run_id), pool)
    agents = [tm, prep, load_agent, solar_agent, dev, integ]

    print(f"[START] mode=research async_interactive run_id={run_id}", flush=True)

    start = asyncio.get_running_loop().time()
    last_seq = pool.get_all()[-1].seq if (args.print_messages and pool.size() > 0) else -1
    handled_gate_seq = -1
    active_gate = None

    while (asyncio.get_running_loop().time() - start) < args.timeout_seconds:
        for a in agents:
            a.tick()

        if args.print_messages:
            msgs = pool.get_all()
            new = [m for m in msgs if m.seq > last_seq]
            for m in new[-200:]:
                print(f"[MSG] seq={m.seq} topic={m.topic} kind={m.kind} sender={m.sender} level={m.level}", flush=True)
            if new:
                last_seq = max(m.seq for m in new)

        # Fail fast on errors
        for topic_type in ("prepare", "train", "train_execute", "evaluate"):
            for m in pool.find_by_topic(topic_type, topic_source=run_id):
                if m.kind == "error":
                    print(f"[FAIL] topic={m.topic} sender={m.sender} payload={m.payload}", flush=True)
                    raise SystemExit(2)

        # Gate handling
        gate_msgs = pool.find_by_topic("gate", topic_source=run_id)
        latest_req = None
        for m in gate_msgs[::-1]:
            if m.kind == "event" and (m.payload or {}).get("event") == "gate_requested":
                latest_req = m
                break

        if latest_req is not None and latest_req.seq > handled_gate_seq and active_gate is None:
            handled_gate_seq = latest_req.seq
            active_gate = latest_req
            prompt = (latest_req.payload or {}).get("prompt", "")
            if prompt:
                print("\n" + prompt, flush=True)

            human_text = await _prompt_line("> ")
            pool.publish(
                Message.new(
                    topic=TopicId("user_feedback", run_id),
                    kind="event",
                    sender="Human",
                    sender_role="human",
                    payload={
                        "event": "user_feedback",
                        "gate_id": (latest_req.payload or {}).get("gate_id"),
                        "stage": (latest_req.payload or {}).get("stage"),
                        "human_text": human_text,
                    },
                    correlation_id=latest_req.correlation_id,
                    run_id=run_id,
                )
            )
            active_gate = None

        integ_msgs = pool.find_by_topic("integrate", topic_source=run_id)
        if any(m.kind == "result" and (m.payload or {}).get("event") in ("joint_report_ready", "final_research_summary_ready") for m in integ_msgs):
            print(f"[OK] Completed. run_id={run_id}", flush=True)
            print(f"Check logs/runs/{run_id}/reports/", flush=True)
            return

        await asyncio.sleep(args.tick_seconds)

    print(f"[WARN] Timeout. run_id={run_id}", flush=True)
    raise SystemExit(1)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


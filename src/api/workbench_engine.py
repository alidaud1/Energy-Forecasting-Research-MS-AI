from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import threading
import time
from typing import Any, Dict, List, Optional

from ..agents.base_agent import AgentConfig
from ..agents.message_pool import MessagePool
from ..agents.preparation_agent import PreparationAgent
from ..agents.load_forecasting_agent import LoadForecastingAgent
from ..agents.solar_forecasting_agent import SolarForecastingAgent
from ..agents.model_developer_agent import ModelDeveloperAgent
from ..agents.forecast_integration_agent import ForecastIntegrationAgent
from ..agents.messages import Message, TopicId
from ..agents.interactive_task_manager_agent import InteractiveResearchConfig, InteractiveTaskManagerAgent
from ..phase_c_precheck import run_phase_c_precheck

try:  # optional
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

def make_run_id() -> str:
    return datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")


@dataclass
class RunRuntime:
    run_id: str
    pool: MessagePool
    agents: list
    status: str = "running"  # running|completed|failed|stopped
    last_error: str = ""
    thread: Optional[threading.Thread] = None


class WorkbenchEngine:
    """
    Phase E1: in-process orchestrator runner so the API can publish user_feedback into the same MessagePool
    that the agents are polling.
    """

    def __init__(self, *, log_path: str = "logs/message_pool.jsonl") -> None:
        self.log_path = log_path
        self._lock = threading.Lock()
        self._runs: Dict[str, RunRuntime] = {}

    def list_runs(self) -> List[Dict[str, str]]:
        with self._lock:
            return [{"run_id": r.run_id, "status": r.status} for r in self._runs.values()]

    def get(self, run_id: str) -> Optional[RunRuntime]:
        with self._lock:
            return self._runs.get(run_id)

    def start_research_interactive(
        self,
        *,
        manifest: str,
        W: int,
        H: int,
        stride: int,
        with_baselines: bool,
        max_train_windows: int,
        max_test_windows: int,
        seed: int,
        epochs: int,
        batch_size: int,
        lr: float,
        patience: int,
        load_model: str,
        solar_model: str,
        use_llm: bool = False,
        timeout_seconds: float = 7200.0,
        tick_seconds: float = 1.0,
        precheck: bool = True,
    ) -> str:
        run_id = make_run_id()
        if load_dotenv is not None:
            # Load .env so OPENAI_API_KEY can be picked up without manual export.
            load_dotenv(override=False)
        pool = MessagePool(log_path=self.log_path, load_existing=True)

        if precheck:
            ok, _report = run_phase_c_precheck(manifest=manifest, tracks=("load", "solar"), check_fractional_weather=True)
            if not ok:
                raise RuntimeError("Phase C precheck failed. Fix data artifacts first.")

        tm = InteractiveTaskManagerAgent(
            AgentConfig(agent_id="InteractiveTaskManagerAgent", role="task_manager", run_id=run_id),
            pool,
            InteractiveResearchConfig(
                manifest_path=manifest,
                tracks=["load", "solar"],
                W=int(W),
                H=int(H),
                stride=int(stride),
                with_baselines=bool(with_baselines),
                max_train_windows=int(max_train_windows),
                max_test_windows=int(max_test_windows),
                seed=int(seed),
                load_model=str(load_model),
                solar_model=str(solar_model),
                use_llm=bool(use_llm),
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                patience=int(patience),
            ),
        )
        prep = PreparationAgent(AgentConfig(agent_id="PreparationAgent", role="preparation", run_id=run_id), pool)
        load_agent = LoadForecastingAgent(AgentConfig(agent_id="LoadForecastingAgent", role="load_forecasting", run_id=run_id), pool)
        solar_agent = SolarForecastingAgent(AgentConfig(agent_id="SolarForecastingAgent", role="solar_forecasting", run_id=run_id), pool)
        dev = ModelDeveloperAgent(AgentConfig(agent_id="ModelDeveloperAgent", role="model_developer", run_id=run_id), pool)
        integ = ForecastIntegrationAgent(AgentConfig(agent_id="ForecastIntegrationAgent", role="forecast_integration", run_id=run_id), pool)
        agents = [tm, prep, load_agent, solar_agent, dev, integ]

        rt = RunRuntime(run_id=run_id, pool=pool, agents=agents)

        def loop() -> None:
            start = time.time()
            try:
                while (time.time() - start) < float(timeout_seconds):
                    # tick agents
                    for a in agents:
                        a.tick()

                    # fail fast on errors for this run
                    for topic_type in ("prepare", "train", "train_execute", "evaluate"):
                        for m in pool.find_by_topic(topic_type, topic_source=run_id):
                            if m.kind == "error":
                                rt.status = "failed"
                                rt.last_error = str((m.payload or {}).get("error") or m.payload)
                                return

                    # done when integration finished
                    integ_msgs = pool.find_by_topic("integrate", topic_source=run_id)
                    if any(
                        m.kind == "result" and (m.payload or {}).get("event") in ("joint_report_ready", "final_research_summary_ready")
                        for m in integ_msgs
                    ):
                        rt.status = "completed"
                        return

                    time.sleep(float(tick_seconds))

                rt.status = "failed"
                rt.last_error = "timeout"
            except Exception as e:
                rt.status = "failed"
                rt.last_error = str(e)

        th = threading.Thread(target=loop, name=f"run_{run_id}", daemon=True)
        rt.thread = th

        with self._lock:
            self._runs[run_id] = rt

        th.start()
        return run_id

    def publish_user_feedback(self, run_id: str, *, text: str, gate_id: str | None = None) -> None:
        rt = self.get(run_id)
        if not rt:
            raise KeyError(f"run_id not found: {run_id}")

        # Attach correlation_id to the latest gate request if possible (helps determinism).
        corr = None
        gate_msgs = rt.pool.find_by_topic("gate", topic_source=run_id)
        for m in reversed(gate_msgs):
            p = m.payload or {}
            if m.kind == "event" and p.get("event") == "gate_requested":
                if gate_id is None or str(p.get("gate_id") or "") == str(gate_id):
                    corr = m.correlation_id
                    if gate_id is None:
                        gate_id = str(p.get("gate_id") or "")
                    stage = str(p.get("stage") or "")
                    break
        else:
            stage = ""

        rt.pool.publish(
            Message.new(
                topic=TopicId("user_feedback", run_id),
                kind="event",
                sender="Human",
                sender_role="human",
                payload={
                    "event": "user_feedback",
                    "gate_id": gate_id or "",
                    "stage": stage,
                    "human_text": text,
                },
                correlation_id=corr,
                run_id=run_id,
            )
        )

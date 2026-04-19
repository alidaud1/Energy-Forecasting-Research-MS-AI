from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .schemas import FeedbackRequest, FeedbackResponse, RunStartRequest, RunStartResponse, SnapshotResponse
from .state_projection import latest_active_gate, latest_artifacts, recent_messages
from .workbench_engine import WorkbenchEngine


app = FastAPI(title="GEFCom Workbench API", version="0.1")
engine = WorkbenchEngine(log_path="logs/message_pool.jsonl")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/api/runs")
def runs() -> Dict[str, Any]:
    return {"runs": engine.list_runs()}


@app.post("/api/runs/start", response_model=RunStartResponse)
def start(req: RunStartRequest) -> RunStartResponse:
    run_id = engine.start_research_interactive(
        manifest=req.manifest,
        W=req.W,
        H=req.H,
        stride=req.stride,
        with_baselines=req.with_baselines,
        max_train_windows=req.max_train_windows,
        max_test_windows=req.max_test_windows,
        seed=req.seed,
        epochs=req.epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        patience=req.patience,
        load_model=req.load_model,
        solar_model=req.solar_model,
    )
    return RunStartResponse(run_id=run_id)


@app.post("/api/runs/{run_id}/feedback", response_model=FeedbackResponse)
def feedback(run_id: str, req: FeedbackRequest) -> FeedbackResponse:
    try:
        engine.publish_user_feedback(run_id, text=req.text, gate_id=req.gate_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="run_id not found")
    return FeedbackResponse(ok=True)


@app.get("/api/runs/{run_id}/snapshot", response_model=SnapshotResponse)
def snapshot(run_id: str) -> SnapshotResponse:
    rt = engine.get(run_id)
    if not rt:
        raise HTTPException(status_code=404, detail="run_id not found")
    msgs = [m.to_dict() for m in rt.pool.get_all()]
    gate = latest_active_gate(msgs, run_id=run_id)
    artifacts = latest_artifacts(msgs, run_id=run_id) or None
    prompt = gate.prompt if gate else None
    active_gate = {"gate_id": gate.gate_id, "stage": gate.stage, "seq": gate.seq} if gate else None
    return SnapshotResponse(
        run_id=run_id,
        status=rt.status,
        active_gate=active_gate,
        prompt=prompt,
        last_error=rt.last_error or None,
        artifacts=artifacts,
        recent_messages=recent_messages(msgs, run_id=run_id, limit=80),
    )


@app.get("/api/runs/{run_id}/events")
async def events(run_id: str, last_seq: int = -1):
    """
    Server-Sent Events stream of messages for a run.
    Client supplies `last_seq` to resume.
    """
    rt = engine.get(run_id)
    if not rt:
        raise HTTPException(status_code=404, detail="run_id not found")

    async def gen():
        nonlocal last_seq
        while True:
            msgs = [m.to_dict() for m in rt.pool.get_all()]
            new = [m for m in msgs if m.get("run_id") == run_id and int(m.get("seq", -1)) > int(last_seq)]
            new.sort(key=lambda x: int(x.get("seq", -1)))
            for m in new:
                last_seq = int(m.get("seq", last_seq))
                payload = json.dumps(m, ensure_ascii=False, default=str)
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


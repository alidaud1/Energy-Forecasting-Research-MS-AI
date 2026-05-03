from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .schemas import FeedbackRequest, FeedbackResponse, RunStartRequest, RunStartResponse, SnapshotResponse
from .state_projection import latest_active_gate, latest_artifacts, recent_messages
from .workbench_engine import WorkbenchEngine


app = FastAPI(title="GEFCom Workbench API", version="0.1")
engine = WorkbenchEngine(log_path="logs/message_pool.jsonl")

# Phase E3: serve visual artifacts (plots/reports) from logs/ via stable URLs.
app.mount("/static", StaticFiles(directory="logs"), name="static")

# Phase E4: allow React dev server to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)


ws_manager = ConnectionManager()


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
        use_llm=req.use_llm,
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
        cursor = rt.pool.cursor_from_seq(int(last_seq))
        while True:
            msgs, cursor = rt.pool.read_from_cursor(cursor, run_id=run_id, max_messages=1000)
            for m in msgs:
                last_seq = int(m.seq)
                payload = json.dumps(m.to_dict(), ensure_ascii=False, default=str)
                yield f"data: {payload}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.websocket("/ws")
async def ws(ws: WebSocket, run_id: str = Query(...), last_seq: int = Query(-1)):
    """
    Phase E4: WebSocket stream of messages for a run.

    Client -> server messages:
      {"type":"feedback","text":"...","gate_id":"...?"}

    Server -> client messages:
      {"type":"message","data": <Message dict>}
    """
    rt = engine.get(run_id)
    if not rt:
        # accept then close with policy violation so clients see a clean WS failure
        await ws.accept()
        await ws.close(code=1008)
        return

    await ws_manager.connect(ws)
    try:
        cursor = rt.pool.cursor_from_seq(int(last_seq))
        while True:
            # Send any new messages
            new_msgs, cursor = rt.pool.read_from_cursor(cursor, run_id=run_id, max_messages=1000)
            for m in new_msgs:
                last_seq = int(m.seq)
                await ws.send_text(json.dumps({"type": "message", "data": m.to_dict()}, ensure_ascii=False, default=str))

            # Receive optional client commands without blocking too long
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.2)
                obj = json.loads(raw)
                if isinstance(obj, dict) and obj.get("type") == "feedback":
                    text = str(obj.get("text") or "").strip()
                    gate_id = obj.get("gate_id")
                    if text:
                        engine.publish_user_feedback(run_id, text=text, gate_id=str(gate_id) if gate_id else None)
                        await ws.send_text(json.dumps({"type": "ack", "ok": True}, ensure_ascii=False))
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


def _topic_type(topic: str) -> str:
    return topic.split("/", 1)[0] if "/" in topic else topic


def _choose_predictions_path(
    msgs: list[dict],
    *,
    run_id: str,
    track: str,
    variant: str,
    prefer_transformer: bool = True,
    model_family: str = "",
) -> Tuple[str, dict]:
    evals = [m for m in msgs if m.get("run_id") == run_id and _topic_type(str(m.get("topic", ""))) == "evaluate"]
    # only evaluation_complete
    rows = [m for m in evals if (m.get("payload") or {}).get("event") == "evaluation_complete" and (m.get("payload") or {}).get("track") == track]
    if variant:
        rows = [m for m in rows if str((m.get("payload") or {}).get("variant") or "") == variant]
    if model_family:
        rows = [m for m in rows if str((m.get("payload") or {}).get("model_family") or "").lower() == model_family.lower()]
    if not rows:
        raise FileNotFoundError("No evaluation results found for this run/track/variant.")

    def score(m: dict) -> Tuple[int, float, float]:
        p = m.get("payload") or {}
        fam = str(p.get("model_family") or "").lower()
        is_transformer = 0 if fam == "transformer" else 1
        if not prefer_transformer:
            is_transformer = 0
        metrics = p.get("metrics") or {}
        rmse = float(metrics.get("rmse", float("inf")))
        mae = float(metrics.get("mae", float("inf")))
        return (is_transformer, rmse, mae)

    best = sorted(rows, key=score)[0]
    p = best.get("payload") or {}
    artifacts = p.get("artifacts") or {}
    pred_path = str(artifacts.get("predictions_path") or "")
    if not pred_path:
        raise FileNotFoundError("Missing predictions_path in evaluation artifact.")
    return pred_path, p


@app.get("/api/runs/{run_id}/predictions")
def predictions(
    run_id: str,
    track: str = Query(..., pattern="^(load|solar)$"),
    variant: str = Query("", description="optional: proposed|human_adjusted"),
    model_family: str = Query("", description="optional: baseline|transformer"),
    prefer_transformer: bool = Query(True, description="when model_family is empty, prefer transformers"),
    horizon: int = Query(1, ge=1, le=24),
    zoneid: int = Query(0, ge=0),
    max_points: int = Query(2000, ge=50, le=20000),
) -> Dict[str, Any]:
    """
    Provide UI-friendly prediction series as JSON for charts (Recharts).
    Chooses best evaluation result for given track/variant, reads its predictions parquet,
    filters to horizon + optional zone, and downsamples to max_points.
    """
    rt = engine.get(run_id)
    if not rt:
        raise HTTPException(status_code=404, detail="run_id not found")

    msgs = [m.to_dict() for m in rt.pool.get_all()]
    pred_path, meta = _choose_predictions_path(
        msgs,
        run_id=run_id,
        track=track,
        variant=variant,
        prefer_transformer=bool(prefer_transformer),
        model_family=str(model_family or ""),
    )
    p = Path(pred_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"predictions parquet not found: {pred_path}")

    df = pd.read_parquet(p)
    df = df[df["horizon"] == int(horizon)].copy()
    if zoneid:
        df = df[df["ZONEID"] == int(zoneid)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)

    # Downsample uniformly for UI
    n = len(df)
    if n > max_points:
        step = max(1, n // max_points)
        df = df.iloc[::step].copy()

    rows = [
        {
            "timestamp": t.isoformat(),
            "y_true": float(yt),
            "y_pred": float(yp),
            "ZONEID": int(z),
            "horizon": int(h),
        }
        for t, yt, yp, z, h in zip(df["timestamp"], df["y_true"], df["y_pred"], df["ZONEID"], df["horizon"])
    ]

    return {"run_id": run_id, "track": track, "variant": meta.get("variant", ""), "model": meta.get("model_name", ""), "model_family": meta.get("model_family", ""), "rows": rows}

# Frontend Guide (React Workbench) — Human-in-the-Loop Controls

This file explains **every UI option** in the Phase E4 React frontend and how it maps to the backend agents/gates.
Assume you are a beginner: you can follow this step-by-step without knowing the code.

## 1) What You Are Running

You typically run two processes:

1. **FastAPI backend** (agents + message pool + WebSocket server)
2. **React frontend** (dashboard)

Backend (from `code_implementation/`):

```powershell
uvicorn src.api.workbench_api:app --host 127.0.0.1 --port 8000
```

Frontend (from `code_implementation/react_workbench/`):

```powershell
npm install
npm run dev
```

Open the URL shown by Vite (usually `http://localhost:5173`).

### Important: WebSocket Requirement

The React dashboard uses a WebSocket at:

`ws://127.0.0.1:8000/ws?run_id=<RUN_ID>&last_seq=-1`

If the backend log shows:

`No supported WebSocket library detected`

Install one of these in your Python venv and restart the backend:

```powershell
pip install "uvicorn[standard]"
# or
pip install websockets
# or
pip install wsproto
```

### Streaming Performance Note (Important)

The backend streams agent messages using a **cursor-based** approach (it does not rescan the entire message history on every tick). This keeps the UI responsive even for long runs with many messages.

## 2) Big Picture: How Human-in-the-Loop Works

The system runs in **stages** with **gates** (checkpoints). When a gate is open, the backend is waiting for you to reply.

Main gates you will see:

- **PREP GATE**: confirm dataset/splits are correct (before training)
- **MODEL GATE**: confirm/modify training plan (epochs, lr, models, etc.)
- **POST-PROCESS GATE**: accept forecasts or apply a simple adjustment (scale/bias) after models run

You interact in two ways:

1. **Gate Modal** (appears automatically when a gate opens)
2. **Chat / Feedback box** (always available)

Your message goes into the backend as `user_feedback/<run_id>`, then the Task Manager translates it into structured actions, and agents proceed.

If you type an invalid post-process command (for example `scale load factor=abc`), the deployment step will **reject it** and publish a `deploy:error` message instead of producing a broken forecast artifact.

## 3) Frontend Controls (Every Option)

### A) Top Header

#### 1) `Refresh`
- Updates the left “Runs” list.
- Use it if you started a run elsewhere and want it to appear.

#### 2) `Start Run`
- Starts a new interactive research run using default settings (hardcoded in the React app).
- After you click it, the page will automatically connect to that run’s WebSocket and start streaming messages.

What happens after clicking Start Run:
- Backend creates a new `run_id`
- Agents begin preparing data
- When preparation is ready, the backend publishes a `gate_requested` message
- The frontend shows the **Gate Modal**

### B) Runs Panel (Left Column)

#### 1) Run selector dropdown
- Lets you pick which run you want to monitor/control.
- Each option shows `run_id (status)`.
- When you select a run, the frontend connects to its WebSocket and starts receiving live updates.

### C) Live Log Console (Left Column)

This is your “system is alive” indicator.

Each line is a message from agents:
- `seq`: an increasing message number (the order the system produced events)
- `topic type`: like `prepare`, `train`, `evaluate`, `gate`, `integrate`
- `sender`: which agent produced it (e.g. `PreparationAgent`, `ModelDeveloperAgent`)
- `kind`: event/command/result/error

If you see messages continuing to increment, the backend is running.

### D) Human Gate Modal (Pop-up)

This is the most important HITL control.

When it appears, it contains:
- `stage`: which gate is open (prep/model/postprocess)
- `gate_id`: unique id for that gate (used so your reply applies to the correct gate)
- `prompt`: a long text block describing what the system is asking you to confirm

Buttons/inputs:

#### 1) Text input
Type a command like:
- `approve`
- `reject`
- `set epochs=5 lr=0.001`
- `run load only`
- `disable baselines`
- `accept` (post-process gate)
- `scale solar factor=0.90`

#### 2) `Send`
Sends what you typed (or sends `approve` if the box is empty).

#### 3) `Approve`
Sends `approve` immediately.

What “Approve” does:
- It tells the system: “continue to the next stage”

What “Reject” does:
- It tells the system: “stop this run” (the run will fail/stop)

### E) View Controls (Right Column)

These control how charts are fetched and displayed.

#### 1) `variant`
This chooses which branch of results you want to see:
- `best available`: show whichever the backend considers best for that track
- `proposed`: show the system’s original plan branch (only exists if you edited the plan)
- `human_adjusted`: show the branch after your change (only exists if you edited the plan)

If you never changed the plan at the Model Gate, you will usually only have “best available”.

#### 2) `horizon`
Forecast horizon step (1..24):
- `1` means “next hour”
- `24` means “24 hours ahead”

Charts display only one horizon at a time to keep the UI fast.

#### 3) `Overlay variants`
If enabled:
- the chart will overlay the *other* variant as a **dashed gray** line
- example: if you selected `human_adjusted`, it will try to also show `proposed`

This only works if your run produced both variants (Phase D3 dual-branch).

#### 4) `Overlay baseline`
If enabled:
- the chart overlays an XGBoost baseline prediction (dashed green)
- your primary prediction is still the transformer (solid blue)

This only works if baselines were enabled for the run (`with_baselines=true`).

#### 5) `Refresh charts`
Fetches fresh chart data from the backend.

Backend endpoint used:
- `/api/runs/<run_id>/predictions?track=load|solar&variant=...&horizon=...`

### E2) Catching Up After Refresh (`last_seq`)

The frontend keeps track of the last message sequence it has rendered (`last_seq`). If you refresh the tab, the frontend reconnects with the latest `last_seq` so it can **catch up** on missed messages without duplicating older ones.

### F) Charts (Load + Solar)

Each chart shows:
- **Actual** (black): `y_true`
- **Primary prediction** (blue): the main series selected by your view controls
- **Other variant** (gray dashed): only if “Overlay variants” is enabled
- **Baseline** (green dashed): only if “Overlay baseline” is enabled

If you don’t see lines:
- the run might still be training
- or the backend can’t find a predictions parquet yet
- or WebSocket isn’t connected

### G) Report Panel

This renders the latest report produced by the integration agent:
- `joint_decision_report.md` or the “final summary”

It is fetched via FastAPI static route:
- backend mounts `logs/` at `/static/`

So when the report says:
`logs/runs/<run_id>/reports/joint_decision_report.md`

The UI fetches:
`http://127.0.0.1:8000/static/runs/<run_id>/reports/joint_decision_report.md`

The UI also scans the markdown for `Plot: ` paths and displays PNG plots automatically.

### G2) Optional SSE Stream (Debug / Fallback)

If WebSockets are blocked in your environment, you can still stream messages via Server-Sent Events:

`GET /api/runs/<run_id>/events?last_seq=-1`

The React app uses WebSockets by default, but this SSE endpoint is useful for debugging with a browser or curl.

### H) Chat / Feedback (Bottom Right Panel)

This is always available and sends a message through WebSocket:
- `{"type":"feedback","text":"...","gate_id":"..."}` (gate id is optional)

How to use it:
- If a gate modal is open, prefer replying in the **gate modal** (it includes the correct `gate_id`).
- You can still use the chat box for “freeform” commands, but if the system is currently waiting at a gate, it’s best to respond to that gate.

## 4) What Messages Can You Send (Examples)

### Prep Gate (data confirmation)
Use:
- `approve`
- `reject`
- `run load only`
- `run solar only`
- `run both`
- `enable baselines`
- `disable baselines`
- `set W=168 H=24 stride=24`

### Model Gate (training plan)
Use:
- `approve`
- `reject`
- `set epochs=5 batch_size=128 lr=0.001 patience=2`
- `set load_model=itransformer solar_model=patchtst`
- `disable baselines`
- `run load only`

### Post-Process Gate (deployment-style)
Use:
- `accept`
- `scale load factor=1.04`
- `scale solar hours=12-16 factor=0.85`
- `bias load add=10`
- `bias solar add=-0.1`
- (optional triggers) `scale load factor=1.05 theta=5000`
- (optional triggers) `scale solar factor=0.90 phi_col=VAR169 phi_theta=200`

If you send a command with invalid numbers (NaN/inf/non-numeric), the deployment operator will publish `deploy:error` and **will not** write adjusted predictions.

## 5) What You Should Expect To See (Normal Flow)

1. Click **Start Run**
2. Watch **Live Log Console**; you should see `prepare/... ready`
3. Gate modal appears: **PREP GATE**
4. Type `approve`
5. Gate modal appears: **MODEL GATE**
6. Optionally type `set epochs=2 lr=0.001`, then `approve`
7. Training starts (console shows train/evaluate events)
8. Gate modal appears: **POST-PROCESS GATE**
9. Type `accept` (or apply `scale/bias`)
10. Report panel shows the final report and plots
11. Use View Controls to overlay baseline / variants and compare results

## 6) Common Issues

### “WebSocket 404 / Unsupported upgrade”
Cause: missing websocket support in Python env.
Fix:
```powershell
pip install "uvicorn[standard]"
```
Restart `uvicorn`.

### Charts show nothing
Most common causes:
- run is still training
- WebSocket not connected (no live messages)
- baseline overlay enabled but baselines were disabled
- variant overlay enabled but you did not create both branches
- you selected a `variant` that does not exist for this run (try “best available”)

### Net-load not computed
Your load dataset (Task 1) and solar dataset (Task 15) do not overlap in time in this setup.
So net-load overlay is not available unless you prepare matched tasks/timelines.

### “I don’t see progress, it looks stuck”
Typical reasons:
- Transformers and XGBoost can take minutes on CPU-only systems.
- Watch the Live Log Console: if `seq` keeps increasing, training is still running.
- If the run times out, increase backend timeout (CLI) or reduce training settings (epochs, max windows).

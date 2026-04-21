# Phase E1 - Unified Agent Workbench (API + UI)

Phase E1 adds a web workbench on top of the existing Phase A–D system (no rewrite of agents).

## What Gets Added

- FastAPI bridge: start runs, send feedback, read snapshots, SSE events.
- Streamlit workbench: chat + gate cards + live monitoring + artifacts/audit view.

## How To Run (Local)

1) Install dependencies in your venv:

```powershell
pip install -r requirements.txt
```

2) Start the FastAPI backend (in `code_implementation/`):

```powershell
uvicorn src.api.workbench_api:app --host 127.0.0.1 --port 8000
```

3) Start the Streamlit UI (in a second terminal, same folder):

```powershell
streamlit run src/visualization/workbench_app.py
```

4) Open the Streamlit URL shown in the terminal, then:

- click “Start Interactive Research Run”
- when a gate appears, type `approve` or `set epochs=5` etc
- monitor artifacts and messages live

## API Endpoints (MVP)

- `GET /api/health`
- `GET /api/runs`
- `POST /api/runs/start`
- `POST /api/runs/{run_id}/feedback`
- `GET /api/runs/{run_id}/snapshot`
- `GET /api/runs/{run_id}/events` (SSE)


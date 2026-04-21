# React Workbench (Phase E4)

This is the React + Tailwind + Recharts frontend for the unified agent workbench.

## Run

1) Start the FastAPI server (from `code_implementation/`):

```powershell
uvicorn src.api.workbench_api:app --host 127.0.0.1 --port 8000
```

2) Start the React dev server (from `code_implementation/react_workbench/`):

```powershell
npm install
npm run dev
```

## Notes

- WebSocket: `ws://127.0.0.1:8000/ws?run_id=<run_id>`
- REST: `/api/runs/start`, `/api/runs/<run_id>/predictions`, `/api/runs/<run_id>/snapshot`
- Artifacts are served by FastAPI at `/static/...` (mounted from `logs/`)


# Phase E3 - OpenAI Task Manager + Artifact Rendering

Phase E3 adds:
- an OpenAI-backed `LLMService` that helps the Task Manager translate free-form human text into safe, structured gate commands (with rule-based fallback)
- basic artifact rendering in the Streamlit workbench (reports + plot PNGs)

## Enable LLM (Task Manager)

1) Put your key in `.env`:

```env
OPENAI_API_KEY=...
```

2) Install deps:

```powershell
pip install -r requirements.txt
```

3) Start the workbench API + UI, and start a run with `use_llm=true` in the UI checkbox.

Notes:
- Default model is pinned: `gpt-5.4-2026-03-05` (reproducible).
- If the LLM is unavailable, the system falls back to the existing rule parser.

## Artifact Rendering

The FastAPI server mounts `logs/` at:
- `GET /static/...`

The Streamlit UI uses this to render:
- `reports/*.md` via `st.markdown`
- `plots/*.png` via `st.image`


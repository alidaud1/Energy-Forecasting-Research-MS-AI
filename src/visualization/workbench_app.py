from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import requests
import streamlit as st


DEFAULT_API = os.environ.get("WORKBENCH_API", "http://127.0.0.1:8000")


def _get(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def _snapshot(run_id: str) -> Dict[str, Any]:
    return _get(f"{DEFAULT_API}/api/runs/{run_id}/snapshot")


st.set_page_config(page_title="Unified Agent Workbench", layout="wide")

st.title("Unified Agent Workbench (Load + Solar Forecasting)")

with st.sidebar:
    st.subheader("Backend")
    api = st.text_input("FastAPI URL", DEFAULT_API)
    if api and api != DEFAULT_API:
        DEFAULT_API = api  # type: ignore
    st.caption("Tip: set env var `WORKBENCH_API` to persist this.")

    st.subheader("Start Run")
    manifest = st.text_input("Manifest", "data/processed/dataset_manifest.json")
    c1, c2, c3 = st.columns(3)
    W = c1.number_input("W", value=168, step=24)
    H = c2.number_input("H", value=24, step=1)
    stride = c3.number_input("stride", value=24, step=1)
    with_baselines = st.checkbox("with_baselines (XGBoost)", value=True)
    epochs = st.number_input("epochs", value=5, step=1)
    batch_size = st.number_input("batch_size", value=128, step=32)
    lr = st.number_input("lr", value=0.001, step=0.0005, format="%.6f")
    patience = st.number_input("patience", value=2, step=1)
    load_model = st.selectbox("load_model", ["patchtst", "itransformer"], index=0)
    solar_model = st.selectbox("solar_model", ["patchtst", "itransformer"], index=0)

    if st.button("Start Interactive Research Run", type="primary"):
        res = _post(
            f"{DEFAULT_API}/api/runs/start",
            {
                "manifest": manifest,
                "W": int(W),
                "H": int(H),
                "stride": int(stride),
                "with_baselines": bool(with_baselines),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "patience": int(patience),
                "load_model": str(load_model),
                "solar_model": str(solar_model),
            },
        )
        st.session_state["run_id"] = res["run_id"]

st.divider()

run_id = st.session_state.get("run_id", "")
colA, colB = st.columns([1, 2])

with colA:
    st.subheader("Run")
    runs = _get(f"{DEFAULT_API}/api/runs").get("runs", [])
    run_ids = [r["run_id"] for r in runs]
    chosen = st.selectbox("Select run_id", [""] + run_ids, index=(run_ids.index(run_id) + 1) if run_id in run_ids else 0)
    if chosen:
        st.session_state["run_id"] = chosen
        run_id = chosen

    if not run_id:
        st.info("Start a run from the sidebar, or select an existing run.")
    else:
        snap = _snapshot(run_id)
        st.write(f"status: `{snap['status']}`")
        if snap.get("last_error"):
            st.error(snap["last_error"])

        gate = snap.get("active_gate")
        prompt = snap.get("prompt") or ""
        if gate:
            st.subheader("Gate")
            st.code(prompt, language="text")
            text = st.text_input("Your command", key="gate_cmd")
            c1, c2 = st.columns(2)
            if c1.button("Send", disabled=not bool(text.strip())):
                _post(f"{DEFAULT_API}/api/runs/{run_id}/feedback", {"text": text.strip(), "gate_id": gate.get("gate_id")})
                st.session_state["gate_cmd"] = ""
            if c2.button("Approve"):
                _post(f"{DEFAULT_API}/api/runs/{run_id}/feedback", {"text": "approve", "gate_id": gate.get("gate_id")})

        st.subheader("Chat / Feedback")
        free = st.text_input("Send feedback (raw)", key="free_feedback")
        if st.button("Send feedback", disabled=not bool(free.strip())):
            _post(f"{DEFAULT_API}/api/runs/{run_id}/feedback", {"text": free.strip()})
            st.session_state["free_feedback"] = ""

with colB:
    st.subheader("Live Monitor")

    if not run_id:
        st.empty()
    else:
        # Light polling: keep it demo-stable across Streamlit versions.
        auto = st.toggle("Auto refresh (2s)", value=True)
        ph = st.empty()

        def render():
            snap = _snapshot(run_id)
            artifacts = snap.get("artifacts") or {}
            integ = (artifacts.get("integration") or {}) if isinstance(artifacts, dict) else {}
            if integ:
                st.markdown("**Artifacts**")
                st.json(integ)
            st.markdown("**Recent messages**")
            st.dataframe(snap.get("recent_messages") or [], use_container_width=True, height=520)

        if auto:
            # crude refresh loop; Streamlit will stop it on interaction.
            render()
            time.sleep(2)
            st.rerun()
        else:
            if st.button("Refresh now"):
                render()
            else:
                render()


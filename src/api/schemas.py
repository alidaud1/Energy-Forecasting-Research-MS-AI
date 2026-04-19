from __future__ import annotations

from pydantic import BaseModel, Field


class RunStartRequest(BaseModel):
    manifest: str = Field(default="data/processed/dataset_manifest.json")
    W: int = 168
    H: int = 24
    stride: int = 24

    with_baselines: bool = True
    max_train_windows: int = 20000
    max_test_windows: int = 5000
    seed: int = 42

    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 10
    load_model: str = Field(default="patchtst")
    solar_model: str = Field(default="patchtst")


class RunStartResponse(BaseModel):
    run_id: str


class FeedbackRequest(BaseModel):
    text: str
    gate_id: str | None = None


class FeedbackResponse(BaseModel):
    ok: bool = True


class SnapshotResponse(BaseModel):
    run_id: str
    status: str
    active_gate: dict | None = None
    prompt: str | None = None
    last_error: str | None = None
    artifacts: dict | None = None
    recent_messages: list[dict] = []


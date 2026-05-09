from __future__ import annotations

from pydantic import BaseModel, Field


class HpoConfig(BaseModel):
    enabled: bool = False
    track: str = Field(default="load", pattern="^(load|solar)$")
    n_trials: int = 10
    init_batch_K: int = 2
    batch_B: int = 2
    constant_liar: bool = True


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

    # Phase E3: enable LLM-backed Task Manager translation (requires OPENAI_API_KEY).
    use_llm: bool = False

    # Phase E5: experiment setup (optional)
    interactive: bool = True
    description: str = ""
    tracks: list[str] = Field(default_factory=lambda: ["load", "solar"])
    models: list[str] = Field(default_factory=list)
    hpo: HpoConfig = Field(default_factory=HpoConfig)


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

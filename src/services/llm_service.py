from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


LLMAction = Literal["approve", "reject", "set", "accept", "scale", "bias", "clarify", "unknown"]


class LLMDirective(BaseModel):
    """
    Strict directive contract for Phase E3.

    This is intentionally narrow: the LLM is only allowed to map human text into the
    existing pipeline's supported gate commands (approve/reject/set, accept/scale/bias).
    """

    action: LLMAction = "unknown"
    gate: Literal["prep", "model", "postprocess", "unknown"] = "unknown"

    # For action=set
    changes: Dict[str, Any] = Field(default_factory=dict)

    # For postprocess actions
    target: Optional[Literal["load", "solar"]] = None
    factor: Optional[float] = None
    add: Optional[float] = None
    hours: Optional[str] = None  # "18-21" or "12,13,14"

    justification: str = ""
    confidence: float = 0.5
    requires_confirmation: bool = False
    clarification_question: str = ""


@dataclass(frozen=True)
class LLMServiceConfig:
    model: str = "gpt-5.4-2026-03-05"  # pinned snapshot for thesis reproducibility
    reasoning_effort: str = "low"  # directive parsing is mostly classification/translation
    timeout_seconds: float = 30.0


class LLMService:
    """
    Phase E3: OpenAI-backed directive translation at the Task Manager boundary.

    Notes:
    - This service should never directly mutate agent state; it only returns a validated directive object.
    - If OPENAI_API_KEY is not set, this service is disabled and callers should fall back to rules.
    """

    def __init__(self, cfg: Optional[LLMServiceConfig] = None) -> None:
        self.cfg = cfg or LLMServiceConfig()
        self.api_key_present = bool(os.environ.get("OPENAI_API_KEY"))

        self._client = None
        if self.api_key_present:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI()
            except Exception:
                # Keep disabled if dependency is missing or client init fails.
                self._client = None

    def enabled(self) -> bool:
        return bool(self._client is not None)

    def _system_prompt(self) -> str:
        return (
            "You are the Task Manager in a human-in-the-loop energy forecasting system.\n"
            "Your ONLY job is to translate the user's message into a strict directive object for the existing pipeline.\n"
            "Hard rules:\n"
            "- Output must match the provided schema exactly.\n"
            "- Do not invent new agent roles, tools, or configuration keys.\n"
            "- If the user wants to proceed, action=approve (or accept at postprocess).\n"
            "- If unclear, action=clarify and ask a short clarification_question.\n"
            "Supported keys for changes (action=set):\n"
            "- tracks: [\"load\"], [\"solar\"], or [\"load\",\"solar\"]\n"
            "- with_baselines: true/false\n"
            "- W, H, stride: integers\n"
            "- epochs, batch_size, lr, patience: numbers\n"
            "- load_model, solar_model: \"patchtst\" or \"itransformer\"\n"
            "Supported postprocess:\n"
            "- scale/bias with target=load|solar, and factor/add, optional hours like \"18-21\".\n"
        )

    def translate(
        self,
        *,
        human_text: str,
        gate: Literal["prep", "model", "postprocess"],
        gate_prompt: str = "",
        pinned_state: Optional[Dict[str, Any]] = None,
        recent_events: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMDirective:
        if not self.enabled():
            return LLMDirective(action="unknown", gate=gate, clarification_question="LLM disabled (missing OPENAI_API_KEY or openai package).")

        pinned_state = pinned_state or {}
        recent_events = recent_events or []

        # Keep memory compact (paper-aligned): pinned summary + a small recent window.
        user_content = {
            "gate": gate,
            "gate_prompt": (gate_prompt or "")[:4000],
            "pinned_state": pinned_state,
            "recent_events": recent_events[-5:],
            "human_text": human_text,
        }

        # Use Structured Outputs via Responses.parse (Pydantic model).
        # Official docs: https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses&lang=python
        try:
            response = self._client.responses.parse(  # type: ignore[union-attr]
                model=self.cfg.model,
                input=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": f"Return JSON only.\n\n{user_content}"},
                ],
                text_format=LLMDirective,
                # Reasoning controls are supported by GPT-5.4 family.
                reasoning={"effort": self.cfg.reasoning_effort},
            )
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                return LLMDirective(action="clarify", gate=gate, clarification_question="I couldn't parse a directive. Please restate as approve/reject/set/scale/bias.")
            # Ensure gate echo
            if parsed.gate == "unknown":
                parsed.gate = gate
            return parsed
        except Exception:
            return LLMDirective(action="clarify", gate=gate, clarification_question="Directive translation failed. Please use approve/reject/set/scale/bias.")


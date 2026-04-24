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
    # Phase 105: threshold triggers (deployment-stage)
    theta: Optional[float] = None  # apply only if y_pred >= theta
    phi_col: Optional[str] = None  # external variable column name (from test/clean parquet)
    phi_theta: Optional[float] = None  # apply only if phi_col >= phi_theta

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
        self._last_usage: Dict[str, Any] = {}
        if self.api_key_present:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI()
            except Exception:
                # Keep disabled if dependency is missing or client init fails.
                self._client = None

    def enabled(self) -> bool:
        return bool(self._client is not None)

    def last_usage(self) -> Dict[str, Any]:
        """
        Best-effort last-call usage. Empty dict when unavailable.
        """
        return dict(self._last_usage or {})

    def last_cost_usd(self) -> Optional[float]:
        """
        Best-effort USD estimate based on env-provided rates.

        To enable, set:
        - OPENAI_USD_PER_1K_INPUT_TOKENS
        - OPENAI_USD_PER_1K_OUTPUT_TOKENS

        Returns None when rates or usage are unavailable.
        """
        usage = self.last_usage()
        if not usage:
            return None
        try:
            in_rate = float(os.environ.get("OPENAI_USD_PER_1K_INPUT_TOKENS", ""))
            out_rate = float(os.environ.get("OPENAI_USD_PER_1K_OUTPUT_TOKENS", ""))
        except Exception:
            return None

        # Support multiple shapes.
        try:
            input_tokens = float(usage.get("input_tokens", usage.get("prompt_tokens", usage.get("input", 0))) or 0)
            output_tokens = float(usage.get("output_tokens", usage.get("completion_tokens", usage.get("output", 0))) or 0)
        except Exception:
            return None

        cost = (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate
        if cost != cost:  # NaN
            return None
        return float(cost)

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
            "- optional threshold triggers: theta (y_pred >= theta), and/or phi_col + phi_theta (phi_col >= phi_theta).\n"
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
            # Best-effort usage extraction (SDK varies by version).
            try:
                usage = getattr(response, "usage", None)
                if usage is None:
                    self._last_usage = {}
                elif isinstance(usage, dict):
                    self._last_usage = dict(usage)
                elif hasattr(usage, "model_dump"):
                    self._last_usage = dict(usage.model_dump())  # type: ignore[attr-defined]
                else:
                    self._last_usage = {"raw": str(usage)}
            except Exception:
                self._last_usage = {}
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                return LLMDirective(action="clarify", gate=gate, clarification_question="I couldn't parse a directive. Please restate as approve/reject/set/scale/bias.")
            # Ensure gate echo
            if parsed.gate == "unknown":
                parsed.gate = gate
            return parsed
        except Exception:
            return LLMDirective(action="clarify", gate=gate, clarification_question="Directive translation failed. Please use approve/reject/set/scale/bias.")

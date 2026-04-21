from __future__ import annotations

import os

from src.services.llm_service import LLMService


def test_llm_service_disabled_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    svc = LLMService()
    assert not svc.enabled()

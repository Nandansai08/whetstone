from __future__ import annotations

import json
import os
import urllib.request

import pytest

# Module-level gate to ensure integration tests only run when explicitly requested
if os.environ.get("WHETSTONE_INTEGRATION_TEST") != "1":
    pytest.skip(
        "Integration tests skipped. Set WHETSTONE_INTEGRATION_TEST=1 to enable.",
        allow_module_level=True,
    )


def _get_ollama_model() -> str | None:
    """Helper to check if Ollama is running and has available models."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3.0) as response:
            data = json.loads(response.read().decode())
            models = data.get("models", [])
            if models:
                return models[0]["name"]
    except Exception:
        pass
    return None


def _run_orchestration_assertions(monkeypatch, model_config) -> None:
    """Helper to patch models, execute orchestration, and verify states."""
    from builder_agent import config
    from builder_agent.orchestrate import orchestrate

    # Monkeypatch all orchestration models to target the tested configuration
    monkeypatch.setattr(config, "WORKER_MODEL", model_config)
    monkeypatch.setattr(config, "JUDGE_MODEL", model_config)
    monkeypatch.setattr(config, "PLANNER_MODEL", model_config)
    monkeypatch.setattr(config, "ESCALATION_MODEL", model_config)

    result = orchestrate(
        "Build a function add(a, b)",
        interactive=False,
        memory=None,
    )

    # Verify stable completion states
    assert result["succeeded"] is True
    assert result["artifact"] is not None
    assert len(result["artifact"].strip()) > 0


def test_openrouter_integration(monkeypatch):
    """Verify end-to-end orchestration using OpenRouter API."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not set.")

    from builder_agent.config import ModelConfig

    openrouter_url = "https://openrouter.ai/api/v1"
    model_id = os.environ.get("OPENROUTER_MODEL_ID", "meta-llama/llama-4-scout")
    model = ModelConfig(
        provider="openai",
        model_id=model_id,
        api_key_env="OPENROUTER_API_KEY",
        base_url=openrouter_url,
    )

    _run_orchestration_assertions(monkeypatch, model)


def test_ollama_integration(monkeypatch):
    """Verify end-to-end orchestration using a local Ollama model."""
    ollama_model = _get_ollama_model()
    if not ollama_model:
        pytest.skip("Ollama local service is not reachable or has no models pulled.")

    from builder_agent.config import ModelConfig

    model = ModelConfig(
        provider="openai",
        model_id=ollama_model,
        base_url="http://localhost:11434/v1",
    )

    _run_orchestration_assertions(monkeypatch, model)

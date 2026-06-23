import json
from unittest.mock import patch

from builder_agent.clarify import clarify


def _fake_llm_response(prompt, *, model, system="", max_tokens=4096):
    return json.dumps({
        "description": "A CLI calculator",
        "acceptance_criteria": [
            "adds two integers",
            "handles negative numbers",
        ],
        "assumptions": ["Python only"],
        "output_type": "python_module",
    })


@patch("builder_agent.clarify.ask", side_effect=_fake_llm_response)
def test_clarify_produces_spec_with_criteria(mock_ask):
    spec = clarify("build a calculator", interactive=False)
    assert spec.request == "build a calculator"
    assert len(spec.acceptance_criteria) >= 1
    assert spec.description == "A CLI calculator"
    assert spec.output_type == "python_module"


@patch("builder_agent.clarify.ask", side_effect=_fake_llm_response)
def test_clarify_non_interactive_passes_defaults_block(mock_ask):
    clarify("build X", interactive=False)
    prompt_arg = mock_ask.call_args[0][0]
    assert "sensible defaults" in prompt_arg


@patch("builder_agent.clarify.ask", side_effect=_fake_llm_response)
def test_clarify_interactive_omits_defaults_block(mock_ask):
    clarify("build X", interactive=True)
    prompt_arg = mock_ask.call_args[0][0]
    assert "sensible defaults" not in prompt_arg

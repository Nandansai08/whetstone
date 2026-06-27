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


def test_detect_ambiguity_no_questions():
    with patch("builder_agent.clarify.ask", return_value="[]") as mock_ask:
        from builder_agent.clarify import detect_ambiguity
        res = detect_ambiguity("simple request")
        assert res == []
        mock_ask.assert_called_once()


def test_detect_ambiguity_with_questions():
    with patch("builder_agent.clarify.ask", return_value='["Q1?", "Q2?"]') as mock_ask:
        from builder_agent.clarify import detect_ambiguity
        res = detect_ambiguity("ambiguous request")
        assert res == ["Q1?", "Q2?"]
        mock_ask.assert_called_once()


def test_detect_ambiguity_caps_at_3():
    val = '["Q1?", "Q2?", "Q3?", "Q4?"]'
    with patch("builder_agent.clarify.ask", return_value=val):
        from builder_agent.clarify import detect_ambiguity
        res = detect_ambiguity("vague request")
        assert res == ["Q1?", "Q2?", "Q3?"]


def test_detect_ambiguity_invalid_json():
    with patch("builder_agent.clarify.ask", return_value='not a json list'):
        from builder_agent.clarify import detect_ambiguity
        res = detect_ambiguity("bad json response")
        assert res == []


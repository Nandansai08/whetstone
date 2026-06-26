from unittest.mock import patch

from builder_agent.generate import generate, self_critique
from builder_agent.schemas import MemoryRecord, Spec, SubTask

SPEC = Spec(
    request="calculator",
    description="A CLI calculator",
    acceptance_criteria=["adds two integers"],
    assumptions=[],
    output_type="python_module",
)
SUBTASK = SubTask(
    id="t1",
    description="implement add function",
    acceptance_criteria=["returns sum of two ints"],
)


def _fake_generate(prompt, *, model, system="", max_tokens=4096):
    return "def add(a, b): return a + b"


def _fake_generate_stream(prompt, *, model, system="", max_tokens=4096):
    yield "def add("
    yield "a, b): "
    yield "return a + b"


def _fake_critique(prompt, *, model, system="", max_tokens=4096):
    return "def add(a, b):\n    return a + b"


@patch("builder_agent.generate.ask", side_effect=_fake_generate)
def test_generate_returns_code(mock_ask):
    code = generate(SUBTASK, SPEC)
    assert "def add" in code
    mock_ask.assert_called_once()


@patch("builder_agent.generate.ask", side_effect=_fake_generate)
def test_generate_threads_feedback(mock_ask):
    generate(SUBTASK, SPEC, feedback="TypeError on line 3")
    prompt_arg = mock_ask.call_args[0][0]
    assert "TypeError on line 3" in prompt_arg
    assert "Previous attempt failed" in prompt_arg


@patch("builder_agent.generate.ask", side_effect=_fake_generate)
def test_generate_no_feedback_block_when_none(mock_ask):
    generate(SUBTASK, SPEC, feedback=None)
    prompt_arg = mock_ask.call_args[0][0]
    assert "Previous attempt failed" not in prompt_arg


@patch("builder_agent.generate.ask", side_effect=_fake_generate)
def test_generate_includes_memory_hints(mock_ask):
    hint = MemoryRecord(
        request="calc",
        output_type="python_module",
        subtask_desc="add function",
        failures=["wrong return"],
        fix_summary="added return statement",
        final_code="def add(a,b): return a+b",
        embedding=[],
    )
    generate(SUBTASK, SPEC, memory_hints=[hint])
    prompt_arg = mock_ask.call_args[0][0]
    assert "added return statement" in prompt_arg
    assert "Hints from similar past builds" in prompt_arg


@patch("builder_agent.generate.ask", side_effect=_fake_generate)
def test_generate_empty_hints_no_hints_block(mock_ask):
    generate(SUBTASK, SPEC, memory_hints=[])
    prompt_arg = mock_ask.call_args[0][0]
    assert "Hints from similar past builds" not in prompt_arg


@patch("builder_agent.generate.ask_stream", side_effect=_fake_generate_stream)
def test_generate_invokes_on_chunk(mock_ask_stream):
    chunks_received = []
    def callback(chunk):
        chunks_received.append(chunk)

    code = generate(SUBTASK, SPEC, on_chunk=callback)
    assert code == "def add(a, b): return a + b"
    assert chunks_received == ["def add(", "a, b): ", "return a + b"]


@patch("builder_agent.generate.ask", side_effect=_fake_critique)
def test_self_critique_alters_code(mock_ask):
    original = "def add(a, b): return a + b"
    improved = self_critique(original, SUBTASK)
    assert improved != original
    assert "def add" in improved
    mock_ask.assert_called_once()


@patch("builder_agent.generate.ask", side_effect=_fake_critique)
def test_self_critique_includes_criteria(mock_ask):
    self_critique("code", SUBTASK)
    prompt_arg = mock_ask.call_args[0][0]
    assert "returns sum of two ints" in prompt_arg

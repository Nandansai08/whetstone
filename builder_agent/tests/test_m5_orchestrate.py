import json
from unittest.mock import patch

from builder_agent.budget import TokenBudget
from builder_agent.config import ModelConfig
from builder_agent.orchestrate import _detect_plateau
from builder_agent.schemas import Spec, SubTask

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

WORKER = ModelConfig("anthropic", "claude-sonnet-4-6")
JUDGE = ModelConfig("openai", "gpt-4o")
ESCALATION = ModelConfig("openai", "gpt-4o-escalation")


def _ask_stream_mock_wrapper(mock_ask):
    def stream_side_effect(*args, **kwargs):
        yield mock_ask(*args, **kwargs)
    return stream_side_effect


# --- _detect_plateau unit tests ---


def test_plateau_not_enough_data():
    assert not _detect_plateau([5], patience=2)
    assert not _detect_plateau([5, 6], patience=2)


def test_plateau_flat_scores():
    assert _detect_plateau([5, 5, 5], patience=2)


def test_plateau_improving_scores():
    assert not _detect_plateau([3, 5, 6], patience=2)


def test_plateau_oscillating():
    # 7, 6, 7, 6 — max(window=[7,6]) = 7, max(before=[7,6]) = 7 → plateau
    assert _detect_plateau([7, 6, 7, 6], patience=2)


def test_plateau_slow_improvement():
    # max(before=[3,5]) = 5, max(window=[5,6]) = 6 > 5 → not plateau
    assert not _detect_plateau([3, 5, 5, 6], patience=2)


# --- Plateau escalation tests ---


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.ESCALATION_MODEL", ESCALATION)
@patch("builder_agent.config.MAX_ITERATIONS", 6)
@patch("builder_agent.config.PLATEAU_PATIENCE", 2)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask")
@patch("builder_agent.verify.ask")
def test_plateau_triggers_escalation(
    mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
    gen_models = []

    def gen_side(prompt, *, model, system="", max_tokens=4096):
        gen_models.append(model)
        return "def add(a,b): return a+b"

    mock_gen_ask.side_effect = gen_side

    scores = iter([
        "assert True", json.dumps({"score": 5, "issues": ["meh"]}),
        "assert True", json.dumps({"score": 5, "issues": ["meh"]}),
        "assert True", json.dumps({"score": 5, "issues": ["meh"]}),
        # After escalation:
        "assert True", json.dumps({"score": 9, "issues": []}),
    ])
    mock_verify_ask.side_effect = lambda p, **kw: next(scores)

    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC)

    assert result["escalated"] is True
    assert result["succeeded"] is True
    # First 3 iters use WORKER, iter 4 uses ESCALATION
    # gen is called twice per iter (generate + self_critique)
    assert gen_models[-1] == ESCALATION
    assert gen_models[-2] == ESCALATION


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.ESCALATION_MODEL", ESCALATION)
@patch("builder_agent.config.MAX_ITERATIONS", 6)
@patch("builder_agent.config.PLATEAU_PATIENCE", 2)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask", return_value="code")
@patch("builder_agent.verify.ask")
def test_plateau_after_escalation_stops(
    mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
    scores = iter([
        "assert True", json.dumps({"score": 5, "issues": ["x"]}),
        "assert True", json.dumps({"score": 5, "issues": ["x"]}),
        "assert True", json.dumps({"score": 5, "issues": ["x"]}),
        # Escalated, still stuck:
        "assert True", json.dumps({"score": 5, "issues": ["x"]}),
        # Should stop here, not continue
        "assert True", json.dumps({"score": 5, "issues": ["x"]}),
    ])
    mock_verify_ask.side_effect = lambda p, **kw: next(scores)

    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC)

    assert result["succeeded"] is False
    assert result["escalated"] is True
    assert result["aborted_reason"] == "plateau"
    assert result["iterations"] == 4


# --- Budget exhaustion tests ---


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.MAX_ITERATIONS", 5)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask", return_value="code")
@patch("builder_agent.verify.ask")
def test_budget_exhaustion_mid_subtask(
    mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
    budget = TokenBudget(limit=100)
    budget.record(50, 60)  # already over

    scores = iter([
        "assert True", json.dumps({"score": 9, "issues": []}),
    ])
    mock_verify_ask.side_effect = lambda p, **kw: next(scores)

    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC, budget=budget)

    assert result["succeeded"] is False
    assert result["aborted_reason"] == "token_budget"
    assert result["iterations"] == 0


def _plan_1_subtask(prompt, *, model, system="", max_tokens=4096):
    return json.dumps([{
        "id": "t1", "description": "add",
        "acceptance_criteria": ["adds"], "depends_on": [],
    }])


def _clarify_resp(prompt, *, model, system="", max_tokens=4096):
    return json.dumps({
        "description": SPEC.description,
        "acceptance_criteria": SPEC.acceptance_criteria,
        "assumptions": [],
        "output_type": "python_module",
    })


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.clarify.ask", side_effect=_clarify_resp)
@patch("builder_agent.plan.ask", side_effect=_plan_1_subtask)
@patch("builder_agent.generate.ask", return_value="def f(): return 1")
@patch("builder_agent.verify.ask")
def test_budget_exhaustion_between_subtasks(
    mock_verify_ask, mock_gen_ask, mock_plan_ask,
    mock_clarify_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
    budget = TokenBudget(limit=100)
    budget.record(50, 60)  # already over

    from builder_agent.orchestrate import orchestrate
    result = orchestrate(
        "build calculator", interactive=False, budget=budget,
    )

    assert result["succeeded"] is False
    assert result["aborted_reason"] == "token_budget"
    assert result["halted_at"] == "t1"
    assert result["usage"] is not None
    assert result["usage"]["total_tokens"] >= 100


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.clarify.ask", side_effect=_clarify_resp)
@patch("builder_agent.plan.ask", side_effect=_plan_1_subtask)
@patch("builder_agent.generate.ask", return_value="def f(): return 1")
@patch("builder_agent.verify.ask")
def test_orchestrate_returns_usage(
    mock_verify_ask, mock_gen_ask, mock_plan_ask,
    mock_clarify_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
    verify_responses = [
        "assert True",
        json.dumps({"score": 9, "issues": []}),
        "assert True",
        json.dumps({"score": 9, "issues": []}),
    ]
    mock_verify_ask.side_effect = iter(verify_responses)

    budget = TokenBudget(limit=999999)

    from builder_agent.orchestrate import orchestrate
    result = orchestrate(
        "build calculator", interactive=False, budget=budget,
    )

    assert result["succeeded"] is True
    assert result["usage"] is not None
    assert "total_tokens" in result["usage"]

import json
from unittest.mock import patch

from builder_agent.schemas import SubTask

SUBTASK = SubTask(
    id="t1",
    description="implement add",
    acceptance_criteria=["returns sum of two ints", "handles negatives"],
)


def _make_test_response(prompt, *, model, system="", max_tokens=4096):
    """Mock for make_tests — returns test code keyed off criteria."""
    if "returns sum of two ints" in prompt:
        return "assert add(1, 2) == 3\nassert add(-1, 1) == 0"
    return "assert True"


def _judge_pass(prompt, *, model, system="", max_tokens=4096):
    return json.dumps({"score": 9, "issues": []})


def _judge_fail(prompt, *, model, system="", max_tokens=4096):
    return json.dumps({"score": 4, "issues": ["poor error handling"]})


@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.verify.ask")
def test_verify_pass_when_tests_and_judge_pass(mock_ask, mock_run):
    mock_ask.side_effect = [
        _make_test_response(
            "returns sum of two ints",
            model=None, system="", max_tokens=4096,
        ),
        _judge_pass("", model=None),
    ]
    from builder_agent.verify import verify
    v = verify(SUBTASK, "def add(a,b): return a+b")
    assert v.passed is True
    assert v.tests_passed is True
    assert v.score == 9


@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.verify.ask")
def test_verify_fail_when_judge_below_threshold(mock_ask, mock_run):
    mock_ask.side_effect = [
        "assert True",
        _judge_fail("", model=None),
    ]
    from builder_agent.verify import verify
    v = verify(SUBTASK, "def add(a,b): return a+b")
    assert v.passed is False
    assert v.tests_passed is True
    assert v.score == 4
    assert len(v.issues) > 0


@patch("builder_agent.verify.run_code")
@patch("builder_agent.verify.ask")
def test_objective_failure_short_circuits_judge(mock_ask, mock_run):
    mock_run.return_value = (False, "AssertionError")
    mock_ask.return_value = "assert False"
    from builder_agent.verify import verify
    v = verify(SUBTASK, "bad code")
    assert v.passed is False
    assert v.tests_passed is False
    assert v.score == 0
    assert mock_ask.call_count == 1  # only make_tests, no judge


@patch("builder_agent.verify.ask")
def test_make_tests_uses_acceptance_criteria(mock_ask):
    mock_ask.side_effect = _make_test_response
    from builder_agent.verify import make_tests
    tests = make_tests(SUBTASK, "def add(a,b): return a+b")
    assert "add(1, 2) == 3" in tests


@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.verify.ask")
def test_worker_and_judge_use_different_models(mock_ask, mock_run):
    models_used = []

    def tracking_ask(prompt, *, model, system="", max_tokens=4096):
        models_used.append(model)
        if len(models_used) == 1:
            return "assert True"
        return json.dumps({"score": 9, "issues": []})

    mock_ask.side_effect = tracking_ask
    from builder_agent.verify import verify
    verify(SUBTASK, "code")
    assert len(models_used) == 2
    assert models_used[0] != models_used[1]

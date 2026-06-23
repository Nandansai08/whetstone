import json
import math
import os
import tempfile
from unittest.mock import patch

from builder_agent.config import ModelConfig
from builder_agent.memory import Memory
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

WORKER = ModelConfig("anthropic", "claude-sonnet-4-6")
JUDGE = ModelConfig("openai", "gpt-4o")


class _StubEmbedder:
    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for i, ch in enumerate(text):
            vec[i % self._dim] += ord(ch) / 1000.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


def _tmp_memory() -> Memory:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return Memory(db_path=path, embedder=_StubEmbedder())


# ---- M2 tests: orchestrate_subtask ----


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask")
@patch("builder_agent.verify.ask")
def test_subtask_exits_early_on_pass(
    mock_verify_ask, mock_gen_ask, mock_run,
):
    mock_gen_ask.return_value = "def add(a,b): return a+b"
    mock_verify_ask.side_effect = [
        "assert True",
        json.dumps({"score": 9, "issues": []}),
    ]
    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC)
    assert result["succeeded"] is True
    assert result["iterations"] == 1


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.MAX_ITERATIONS", 3)
@patch("builder_agent.verify.run_code", return_value=(False, "AssertionError"))
@patch("builder_agent.generate.ask", return_value="bad code")
@patch("builder_agent.verify.ask", return_value="assert False")
def test_subtask_respects_max_iterations(
    mock_verify_ask, mock_gen_ask, mock_run,
):
    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC)
    assert result["succeeded"] is False
    assert result["iterations"] == 3
    assert result["attempt"] is not None


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.MAX_ITERATIONS", 3)
@patch("builder_agent.verify.run_code")
@patch("builder_agent.generate.ask")
@patch("builder_agent.verify.ask")
def test_subtask_feeds_issues_as_feedback(
    mock_verify_ask, mock_gen_ask, mock_run,
):
    call_count = [0]

    def gen_side_effect(prompt, *, model, system="", max_tokens=4096):
        call_count[0] += 1
        return f"code_v{call_count[0]}"

    mock_gen_ask.side_effect = gen_side_effect
    verify_responses = iter([
        "assert False",
        "assert True",
        json.dumps({"score": 9, "issues": []}),
    ])
    mock_verify_ask.side_effect = lambda p, **kw: next(verify_responses)
    mock_run.side_effect = [
        (False, "NameError: x not defined"),
        (True, "ok"),
    ]

    from builder_agent.orchestrate import orchestrate_subtask
    orchestrate_subtask(SUBTASK, SPEC)

    gen_calls = mock_gen_ask.call_args_list
    iter1_gen_prompt = gen_calls[2][0][0]
    assert "NameError" in iter1_gen_prompt


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.MAX_ITERATIONS", 3)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask", return_value="code")
@patch("builder_agent.verify.ask")
def test_subtask_best_so_far_tracks_highest(
    mock_verify_ask, mock_gen_ask, mock_run,
):
    scores = iter([
        "assert True", json.dumps({"score": 5, "issues": ["meh"]}),
        "assert True", json.dumps({"score": 7, "issues": ["ok"]}),
        "assert True", json.dumps({"score": 6, "issues": ["eh"]}),
    ])
    mock_verify_ask.side_effect = lambda p, **kw: next(scores)

    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC)
    assert result["succeeded"] is False
    assert result["attempt"].verdict.score == 7


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask")
@patch("builder_agent.verify.ask")
def test_subtask_worker_judge_different_providers(
    mock_verify_ask, mock_gen_ask, mock_run,
):
    models_seen = []

    def track_gen(prompt, *, model, system="", max_tokens=4096):
        models_seen.append(("gen", model))
        return "code"

    def track_verify(prompt, *, model, system="", max_tokens=4096):
        models_seen.append(("verify", model))
        if model.provider == "openai":
            return json.dumps({"score": 9, "issues": []})
        return "assert True"

    mock_gen_ask.side_effect = track_gen
    mock_verify_ask.side_effect = track_verify

    from builder_agent.orchestrate import orchestrate_subtask
    orchestrate_subtask(SUBTASK, SPEC)

    gen_providers = {m.provider for _, m in models_seen if _ == "gen"}
    judge_providers = {
        m.provider for _, m in models_seen
        if _ == "verify" and m.provider == "openai"
    }
    assert "anthropic" in gen_providers
    assert "openai" in judge_providers


# ---- M3 tests: memory integration ----


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask", return_value="code")
@patch("builder_agent.verify.ask")
@patch("builder_agent.orchestrate.ask", return_value="fixed return value")
def test_subtask_stores_memory_on_pass(
    mock_orch_ask, mock_verify_ask, mock_gen_ask, mock_run,
):
    mock_verify_ask.side_effect = [
        "assert True",
        json.dumps({"score": 9, "issues": []}),
    ]
    mem = _tmp_memory()
    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC, memory=mem)
    assert result["succeeded"] is True
    records = mem.retrieve("implement add function", k=10)
    assert len(records) == 1


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask", return_value="code")
@patch("builder_agent.verify.ask")
def test_subtask_no_memory_works_like_m2(
    mock_verify_ask, mock_gen_ask, mock_run,
):
    mock_verify_ask.side_effect = [
        "assert True",
        json.dumps({"score": 9, "issues": []}),
    ]
    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC, memory=None)
    assert result["succeeded"] is True


# ---- M4 tests: multi-subtask orchestrate ----

MULTI_SPEC = Spec(
    request="calculator",
    description="A CLI calculator",
    acceptance_criteria=["adds two integers", "subtracts", "multiplies"],
    assumptions=[],
    output_type="python_module",
)


def _plan_3_subtasks(prompt, *, model, system="", max_tokens=4096):
    return json.dumps([
        {
            "id": "t1", "description": "add",
            "acceptance_criteria": ["adds"], "depends_on": [],
        },
        {
            "id": "t2", "description": "sub",
            "acceptance_criteria": ["subs"], "depends_on": ["t1"],
        },
        {
            "id": "t3", "description": "mul",
            "acceptance_criteria": ["muls"], "depends_on": ["t1"],
        },
    ])


def _clarify_response(prompt, *, model, system="", max_tokens=4096):
    return json.dumps({
        "description": MULTI_SPEC.description,
        "acceptance_criteria": MULTI_SPEC.acceptance_criteria,
        "assumptions": [],
        "output_type": "python_module",
    })


def _passing_gen(prompt, *, model, system="", max_tokens=4096):
    return "def f(): return 1"


def _passing_verify(prompt, *, model, system="", max_tokens=4096):
    if "score" in (system or "").lower() or "judge" in (system or "").lower():
        return json.dumps({"score": 9, "issues": []})
    return "assert True"


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.clarify.ask", side_effect=_clarify_response)
@patch("builder_agent.plan.ask", side_effect=_plan_3_subtasks)
@patch("builder_agent.generate.ask", side_effect=_passing_gen)
@patch("builder_agent.verify.ask")
def test_orchestrate_runs_all_subtasks_integrates_verifies(
    mock_verify_ask, mock_gen_ask, mock_plan_ask,
    mock_clarify_ask, mock_run,
):
    verify_responses = []
    # 3 subtasks * (make_tests + judge) + final verify (make_tests + judge)
    for _ in range(4):
        verify_responses.append("assert True")
        verify_responses.append(json.dumps({"score": 9, "issues": []}))
    mock_verify_ask.side_effect = iter(verify_responses)

    from builder_agent.orchestrate import orchestrate
    result = orchestrate("build calculator", interactive=False)

    assert result["succeeded"] is True
    assert result["halted_at"] is None
    assert result["artifact"] is not None
    assert result["final_verdict"] is not None
    assert result["final_verdict"].passed is True
    assert len(result["subtask_results"]) == 3


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.MAX_ITERATIONS", 2)
@patch("builder_agent.verify.run_code", return_value=(False, "Error"))
@patch("builder_agent.clarify.ask", side_effect=_clarify_response)
@patch("builder_agent.plan.ask", side_effect=_plan_3_subtasks)
@patch("builder_agent.generate.ask", return_value="bad")
@patch("builder_agent.verify.ask", return_value="assert False")
def test_orchestrate_halts_on_failed_subtask(
    mock_verify_ask, mock_gen_ask, mock_plan_ask,
    mock_clarify_ask, mock_run,
):
    from builder_agent.orchestrate import orchestrate
    result = orchestrate("build calculator", interactive=False)

    assert result["succeeded"] is False
    assert result["halted_at"] == "t1"
    assert result["artifact"] is None
    assert result["final_verdict"] is None


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.clarify.ask", side_effect=_clarify_response)
@patch("builder_agent.plan.ask", side_effect=_plan_3_subtasks)
@patch("builder_agent.generate.ask", side_effect=_passing_gen)
@patch("builder_agent.verify.ask")
@patch("builder_agent.orchestrate.ask", return_value="fix summary")
def test_orchestrate_stores_subtask_and_plan_records(
    mock_orch_ask, mock_verify_ask, mock_gen_ask,
    mock_plan_ask, mock_clarify_ask, mock_run,
):
    verify_responses = []
    for _ in range(4):
        verify_responses.append("assert True")
        verify_responses.append(json.dumps({"score": 9, "issues": []}))
    mock_verify_ask.side_effect = iter(verify_responses)

    mem = _tmp_memory()
    from builder_agent.orchestrate import orchestrate
    orchestrate("build calculator", interactive=False, memory=mem)

    subtask_recs = mem.retrieve(
        "calculator", k=20, record_type="subtask"
    )
    plan_recs = mem.retrieve(
        "calculator", k=20, record_type="plan"
    )
    assert len(subtask_recs) == 3
    assert len(plan_recs) == 1


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.clarify.ask", side_effect=_clarify_response)
@patch("builder_agent.plan.ask", side_effect=_plan_3_subtasks)
@patch("builder_agent.generate.ask", side_effect=_passing_gen)
@patch("builder_agent.verify.ask")
def test_final_verify_uses_spec_criteria(
    mock_verify_ask, mock_gen_ask, mock_plan_ask,
    mock_clarify_ask, mock_run,
):
    verify_calls = []

    def track_verify(prompt, *, model, system="", max_tokens=4096):
        verify_calls.append(prompt)
        if "score" in (system or "").lower() or "judge" in (system or "").lower():
            return json.dumps({"score": 9, "issues": []})
        return "assert True"

    mock_verify_ask.side_effect = track_verify

    from builder_agent.orchestrate import orchestrate
    orchestrate("build calculator", interactive=False)

    # Final verify calls include the spec-level criteria
    final_calls = verify_calls[-2:]  # make_tests + judge for final
    all_text = " ".join(final_calls)
    assert "adds two integers" in all_text
    assert "subtracts" in all_text
    assert "multiplies" in all_text


@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask", return_value="code")
@patch("builder_agent.verify.ask")
def test_retrieve_plan_type_filter(
    mock_verify_ask, mock_gen_ask, mock_run,
):
    mem = _tmp_memory()
    emb = _StubEmbedder()

    mem.store(MemoryRecord(
        request="calc",
        output_type="python_module",
        subtask_desc="add func",
        failures=[],
        fix_summary="ok",
        final_code="code",
        embedding=emb.embed("calc add func"),
        record_type="subtask",
    ))
    mem.store(MemoryRecord(
        request="calc",
        output_type="python_module",
        subtask_desc="A -> B plan",
        failures=[],
        fix_summary="passed",
        final_code="",
        embedding=emb.embed("calc"),
        record_type="plan",
    ))

    plan_only = mem.retrieve("calc", k=10, record_type="plan")
    subtask_only = mem.retrieve("calc", k=10, record_type="subtask")

    assert all(r.record_type == "plan" for r in plan_only)
    assert all(r.record_type == "subtask" for r in subtask_only)

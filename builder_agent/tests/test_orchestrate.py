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


def _ask_stream_mock_wrapper(mock_ask):
    def stream_side_effect(*args, **kwargs):
        yield mock_ask(*args, **kwargs)
    return stream_side_effect


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


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask")
@patch("builder_agent.verify.ask")
def test_subtask_exits_early_on_pass(
    mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
    mock_gen_ask.return_value = "def add(a,b): return a+b"
    mock_verify_ask.side_effect = [
        "assert True",
        json.dumps({"score": 9, "issues": []}),
    ]
    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC)
    assert result["succeeded"] is True
    assert result["iterations"] == 1


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.MAX_ITERATIONS", 3)
@patch("builder_agent.verify.run_code", return_value=(False, "AssertionError"))
@patch("builder_agent.generate.ask", return_value="bad code")
@patch("builder_agent.verify.ask", return_value="assert False")
def test_subtask_respects_max_iterations(
    mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
    from builder_agent.orchestrate import orchestrate_subtask
    result = orchestrate_subtask(SUBTASK, SPEC)
    assert result["succeeded"] is False
    assert result["iterations"] == 3
    assert result["attempt"] is not None


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.MAX_ITERATIONS", 3)
@patch("builder_agent.verify.run_code")
@patch("builder_agent.generate.ask")
@patch("builder_agent.verify.ask")
def test_subtask_feeds_issues_as_feedback(
    mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
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


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.config.MAX_ITERATIONS", 3)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask", return_value="code")
@patch("builder_agent.verify.ask")
def test_subtask_best_so_far_tracks_highest(
    mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
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


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask")
@patch("builder_agent.verify.ask")
def test_subtask_worker_judge_different_providers(
    mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
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


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask", return_value="code")
@patch("builder_agent.verify.ask")
@patch("builder_agent.orchestrate.ask", return_value="fixed return value")
def test_subtask_stores_memory_on_pass(
    mock_orch_ask, mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
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


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.generate.ask", return_value="code")
@patch("builder_agent.verify.ask")
def test_subtask_no_memory_works_like_m2(
    mock_verify_ask, mock_gen_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
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


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.clarify.ask", side_effect=_clarify_response)
@patch("builder_agent.plan.ask", side_effect=_plan_3_subtasks)
@patch("builder_agent.generate.ask", side_effect=_passing_gen)
@patch("builder_agent.verify.ask")
def test_orchestrate_runs_all_subtasks_integrates_verifies(
    mock_verify_ask, mock_gen_ask, mock_plan_ask,
    mock_clarify_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
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


@patch("builder_agent.generate.ask_stream")
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
    mock_clarify_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
    from builder_agent.orchestrate import orchestrate
    result = orchestrate("build calculator", interactive=False)

    assert result["succeeded"] is False
    assert result["halted_at"] == "t1"
    assert result["artifact"] is None
    assert result["final_verdict"] is None


@patch("builder_agent.generate.ask_stream")
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
    mock_plan_ask, mock_clarify_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
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


@patch("builder_agent.generate.ask_stream")
@patch("builder_agent.config.WORKER_MODEL", WORKER)
@patch("builder_agent.config.JUDGE_MODEL", JUDGE)
@patch("builder_agent.verify.run_code", return_value=(True, "ok"))
@patch("builder_agent.clarify.ask", side_effect=_clarify_response)
@patch("builder_agent.plan.ask", side_effect=_plan_3_subtasks)
@patch("builder_agent.generate.ask", side_effect=_passing_gen)
@patch("builder_agent.verify.ask")
def test_final_verify_uses_spec_criteria(
    mock_verify_ask, mock_gen_ask, mock_plan_ask,
    mock_clarify_ask, mock_run, mock_gen_ask_stream,
):
    mock_gen_ask_stream.side_effect = _ask_stream_mock_wrapper(mock_gen_ask)
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


def test_parallel_scheduling_and_dependencies():
    import threading
    import time
    from unittest.mock import MagicMock, patch

    from builder_agent.orchestrate import orchestrate
    from builder_agent.schemas import Plan, SubTask

    plan = Plan(subtasks=[
        SubTask(
            id="t1",
            description="task 1",
            acceptance_criteria=["c1"],
            depends_on=[],
        ),
        SubTask(
            id="t2",
            description="task 2",
            acceptance_criteria=["c2"],
            depends_on=["t1"],
        ),
        SubTask(
            id="t3",
            description="task 3",
            acceptance_criteria=["c3"],
            depends_on=[],
        ),
    ])

    history = []
    lock = threading.Lock()

    def mock_orchestrate_subtask(subtask, *args, **kwargs):
        with lock:
            history.append((subtask.id, "start"))

        if subtask.id == "t1":
            time.sleep(0.1)
        elif subtask.id == "t3":
            time.sleep(0.2)
        elif subtask.id == "t2":
            time.sleep(0.05)

        with lock:
            history.append((subtask.id, "end"))

        return {
            "succeeded": True,
            "attempt": MagicMock(code="code"),
            "iterations": 1,
            "escalated": False,
            "aborted_reason": None,
        }

    with patch("builder_agent.orchestrate.clarify", return_value=SPEC), \
         patch("builder_agent.orchestrate.make_plan", return_value=plan), \
         patch(
             "builder_agent.orchestrate.orchestrate_subtask",
             side_effect=mock_orchestrate_subtask,
         ), \
         patch("builder_agent.orchestrate.integrate", return_value="integrated"), \
         patch("builder_agent.orchestrate.verify") as mock_verify:

         mock_verify_verdict = MagicMock()
         mock_verify_verdict.passed = True
         mock_verify.return_value = mock_verify_verdict

         result = orchestrate("test parallel", interactive=False)

         assert result["succeeded"] is True

         t1_end_idx = history.index(("t1", "end"))
         t3_start_idx = history.index(("t3", "start"))
         t2_start_idx = history.index(("t2", "start"))

         # t2 depends on t1, so t2 must start after t1 ends
         assert t2_start_idx > t1_end_idx

         # t3 is independent of t1, so t3 should start before t1 ends
         assert t3_start_idx < t1_end_idx


def test_progress_callback_metadata():
    from unittest.mock import MagicMock, patch

    from builder_agent.orchestrate import orchestrate
    from builder_agent.schemas import Plan, SubTask

    plan = Plan(subtasks=[
        SubTask(
            id="t1",
            description="task 1",
            acceptance_criteria=["c1"],
            depends_on=[],
        ),
    ])

    events = []
    def progress_cb(event, data):
        events.append((event, data.copy()))

    def mock_orchestrate_subtask(subtask, *args, **kwargs):
        kwargs["on_progress"]("generating", {"subtask": subtask.id, "iteration": 1})
        return {
            "succeeded": True,
            "attempt": MagicMock(code="code"),
            "iterations": 1,
            "escalated": False,
            "aborted_reason": None,
        }

    with patch("builder_agent.orchestrate.clarify", return_value=SPEC), \
         patch("builder_agent.orchestrate.make_plan", return_value=plan), \
         patch(
             "builder_agent.orchestrate.orchestrate_subtask",
             side_effect=mock_orchestrate_subtask,
         ), \
         patch("builder_agent.orchestrate.integrate", return_value="integrated"), \
         patch("builder_agent.orchestrate.verify") as mock_verify:

         mock_verify_verdict = MagicMock()
         mock_verify_verdict.passed = True
         mock_verify.return_value = mock_verify_verdict

         orchestrate("test progress", interactive=False, on_progress=progress_cb)

         start_event = next(e for e in events if e[0] == "subtask_start")
         done_event = next(e for e in events if e[0] == "subtask_done")
         generating_event = next(e for e in events if e[0] == "generating")

         assert start_event[1]["subtask"] == "t1"
         assert start_event[1]["index"] == 0
         assert start_event[1]["total"] == 1

         assert done_event[1]["subtask"] == "t1"
         assert done_event[1]["succeeded"] is True
         assert done_event[1]["index"] == 0
         assert done_event[1]["total"] == 1

         assert generating_event[1]["subtask"] == "t1"
         assert generating_event[1]["iteration"] == 1


def test_sqlite_db_lock():
    import threading
    import time
    from unittest.mock import MagicMock

    from builder_agent.orchestrate import _store_subtask_memory

    memory = MagicMock()
    spec = MagicMock()
    subtask = MagicMock()
    attempts = [MagicMock()]
    best = MagicMock()

    call_times = []

    def mock_store(record):
        call_times.append(time.time())
        time.sleep(0.05)
        call_times.append(time.time())

    memory.store = mock_store

    t1 = threading.Thread(
        target=_store_subtask_memory,
        args=(memory, spec, subtask, attempts, best),
    )
    t2 = threading.Thread(
        target=_store_subtask_memory,
        args=(memory, spec, subtask, attempts, best),
    )

    t1.start()
    time.sleep(0.01)
    t2.start()

    t1.join()
    t2.join()

    # Thread safety check: start of t2 (call_times[2]) >= end of t1 (call_times[1])
    assert call_times[2] >= call_times[1]


def test_budget_exceeded_concurrency():
    from unittest.mock import MagicMock, patch

    from builder_agent.budget import TokenBudget
    from builder_agent.orchestrate import orchestrate
    from builder_agent.schemas import Plan, SubTask

    plan = Plan(subtasks=[
        SubTask(
            id="t1",
            description="task 1",
            acceptance_criteria=["c1"],
            depends_on=[],
        ),
        SubTask(
            id="t2",
            description="task 2",
            acceptance_criteria=["c2"],
            depends_on=["t1"],
        ),
    ])

    budget = TokenBudget(limit=10)

    def mock_orchestrate_subtask(subtask, *args, **kwargs):
        budget.record(5, 6)
        return {
            "succeeded": True,
            "attempt": MagicMock(code="code"),
            "iterations": 1,
            "escalated": False,
            "aborted_reason": None,
        }

    with patch("builder_agent.orchestrate.clarify", return_value=SPEC), \
         patch("builder_agent.orchestrate.make_plan", return_value=plan), \
         patch(
             "builder_agent.orchestrate.orchestrate_subtask",
             side_effect=mock_orchestrate_subtask,
         ), \
         patch("builder_agent.orchestrate.integrate", return_value="integrated"), \
         patch("builder_agent.orchestrate.verify") as mock_verify:

         mock_verify_verdict = MagicMock()
         mock_verify_verdict.passed = True
         mock_verify.return_value = mock_verify_verdict

         res = orchestrate("test budget", interactive=False, budget=budget)
         assert res["succeeded"] is False


def test_scheduler_stops_on_failure():
    import time
    from unittest.mock import MagicMock, patch

    from builder_agent.orchestrate import orchestrate
    from builder_agent.schemas import Plan, SubTask

    # t1 and t3 are independent. t2 depends on t1.
    plan = Plan(subtasks=[
        SubTask(
            id="t1",
            description="task 1",
            acceptance_criteria=["c1"],
            depends_on=[],
        ),
        SubTask(
            id="t2",
            description="task 2",
            acceptance_criteria=["c2"],
            depends_on=["t1"],
        ),
        SubTask(
            id="t3",
            description="task 3",
            acceptance_criteria=["c3"],
            depends_on=[],
        ),
    ])

    history = []

    def mock_orchestrate_subtask(subtask, *args, **kwargs):
        history.append((subtask.id, "start"))
        if subtask.id == "t3":
            # t3 fails immediately
            history.append((subtask.id, "end"))
            return {
                "succeeded": False,
                "attempt": MagicMock(code="code"),
                "iterations": 1,
                "escalated": False,
                "aborted_reason": "failed_t3",
            }
        elif subtask.id == "t1":
            # t1 succeeds after a small delay to ensure t3 fails first
            time.sleep(0.1)
            history.append((subtask.id, "end"))
            return {
                "succeeded": True,
                "attempt": MagicMock(code="code"),
                "iterations": 1,
                "escalated": False,
                "aborted_reason": None,
            }
        elif subtask.id == "t2":
            history.append((subtask.id, "end"))
            return {
                "succeeded": True,
                "attempt": MagicMock(code="code"),
                "iterations": 1,
                "escalated": False,
                "aborted_reason": None,
            }

    with patch("builder_agent.orchestrate.clarify", return_value=SPEC), \
         patch("builder_agent.orchestrate.make_plan", return_value=plan), \
         patch(
             "builder_agent.orchestrate.orchestrate_subtask",
             side_effect=mock_orchestrate_subtask,
         ), \
         patch("builder_agent.orchestrate.integrate", return_value="integrated"), \
         patch("builder_agent.orchestrate.verify") as mock_verify:

         mock_verify_verdict = MagicMock()
         mock_verify_verdict.passed = True
         mock_verify.return_value = mock_verify_verdict

         result = orchestrate("test failure stop", interactive=False)

         # The orchestration should fail because t3 failed
         assert result["succeeded"] is False

         # t1 and t3 should have started and ended
         assert ("t1", "start") in history
         assert ("t1", "end") in history
         assert ("t3", "start") in history
         assert ("t3", "end") in history

         # t2 should NOT have started because t3 failed before t1 finished to unblock t2
         assert ("t2", "start") not in history
def test_orchestrate_subtask_emits_chunks():
    events = []

    def on_progress(event, data):
        events.append((event, data))

    verify_responses = iter([
        "assert True",
        json.dumps({"score": 9, "issues": []}),
    ])

    from unittest.mock import patch

    from builder_agent.orchestrate import orchestrate_subtask

    chunks_gen = (c for c in ["chunk1", "chunk2"])

    with patch("builder_agent.verify.run_code", return_value=(True, "ok")), \
         patch("builder_agent.generate.ask_stream", return_value=chunks_gen), \
         patch("builder_agent.generate.ask", return_value="chunk1chunk2"), \
         patch("builder_agent.verify.ask",
               side_effect=lambda p, **kw: next(verify_responses)):

        orchestrate_subtask(SUBTASK, SPEC, on_progress=on_progress)

    chunk_events = [e for e in events if e[0] == "chunk"]
    assert len(chunk_events) == 2
    assert chunk_events[0][1]["chunk"] == "chunk1"
    assert chunk_events[1][1]["chunk"] == "chunk2"
    assert chunk_events[0][1]["subtask"] == SUBTASK.id

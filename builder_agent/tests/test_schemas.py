from builder_agent.schemas import (
    Attempt,
    MemoryRecord,
    Plan,
    Spec,
    SubTask,
    Verdict,
)


def test_spec_round_trip():
    s = Spec(
        request="build a calculator",
        description="A CLI calculator",
        acceptance_criteria=["adds two numbers"],
        assumptions=["Python only"],
        output_type="python_module",
    )
    assert s.request == "build a calculator"
    assert s.output_type == "python_module"
    assert len(s.acceptance_criteria) == 1


def test_subtask_defaults():
    st = SubTask(id="t1", description="do thing", acceptance_criteria=["works"])
    assert st.depends_on == []


def test_subtask_with_deps():
    st = SubTask(
        id="t2", description="next", acceptance_criteria=["ok"], depends_on=["t1"]
    )
    assert st.depends_on == ["t1"]


def test_plan_holds_subtasks():
    t1 = SubTask(id="t1", description="first", acceptance_criteria=["done"])
    t2 = SubTask(
        id="t2", description="second",
        acceptance_criteria=["done"], depends_on=["t1"],
    )
    p = Plan(subtasks=[t1, t2])
    assert len(p.subtasks) == 2
    assert p.subtasks[1].depends_on == ["t1"]


def test_verdict_fields():
    v = Verdict(passed=True, score=9, tests_passed=True, issues=[], exec_output="ok")
    assert v.passed is True
    assert v.score == 9


def test_verdict_failure():
    v = Verdict(
        passed=False,
        score=3,
        tests_passed=False,
        issues=["syntax error"],
        exec_output="SyntaxError",
    )
    assert v.passed is False
    assert len(v.issues) == 1


def test_attempt_holds_verdict():
    v = Verdict(passed=True, score=8, tests_passed=True, issues=[], exec_output="")
    a = Attempt(iteration=0, code="print(1)", verdict=v)
    assert a.iteration == 0
    assert a.verdict.passed is True


def test_memory_record_fields():
    mr = MemoryRecord(
        request="build X",
        output_type="python_module",
        subtask_desc="do Y",
        failures=["wrong output"],
        fix_summary="added return",
        final_code="def f(): return 1",
        embedding=[0.1, 0.2, 0.3],
    )
    assert len(mr.embedding) == 3
    assert mr.fix_summary == "added return"

from builder_agent import checkpoint
from builder_agent.schemas import Plan, Spec, SubTask

SPEC = Spec(
    request="calculator",
    description="A CLI calculator",
    acceptance_criteria=["adds two integers"],
    assumptions=[],
    output_type="python_module",
)
PLAN = Plan(subtasks=[
    SubTask(id="t1", description="implement add", acceptance_criteria=["a"]),
    SubTask(
        id="t2", description="implement sub",
        acceptance_criteria=["b"], depends_on=["t1"],
    ),
])


def test_build_id_is_stable_and_request_specific():
    assert checkpoint.build_id("foo") == checkpoint.build_id("foo")
    assert checkpoint.build_id("foo") != checkpoint.build_id("bar")


def test_load_missing_returns_none():
    assert checkpoint.load("nonexistent") is None


def test_save_and_load_roundtrip():
    bid = checkpoint.build_id("calculator")
    outputs = {"t1": "def add(a, b): return a + b"}
    completed = {"t1"}

    checkpoint.save(bid, SPEC, PLAN, outputs, completed)
    loaded = checkpoint.load(bid)

    assert loaded is not None
    assert loaded["spec"] == SPEC
    assert loaded["plan"] == PLAN
    assert loaded["outputs"] == outputs
    assert loaded["completed_ids"] == completed


def test_clear_removes_checkpoint():
    bid = checkpoint.build_id("clear me")
    checkpoint.save(bid, SPEC, PLAN, {}, set())
    assert checkpoint.load(bid) is not None

    checkpoint.clear(bid)
    assert checkpoint.load(bid) is None

    checkpoint.clear(bid)  # idempotent, no error on missing file

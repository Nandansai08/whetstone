import json
import math
import os
import tempfile
from unittest.mock import patch

from builder_agent.memory import Memory
from builder_agent.plan import _topo_sort, plan
from builder_agent.schemas import MemoryRecord, Spec, SubTask

SPEC = Spec(
    request="build a calculator",
    description="A CLI calculator with add, subtract, and multiply",
    acceptance_criteria=[
        "adds two integers",
        "subtracts two integers",
        "multiplies two integers",
    ],
    assumptions=["Python only"],
    output_type="python_module",
)

SIMPLE_SPEC = Spec(
    request="hello world",
    description="Print hello world",
    acceptance_criteria=["prints hello world"],
    assumptions=[],
    output_type="python_module",
)


def _single_subtask_response(prompt, *, model, system="", max_tokens=4096):
    return json.dumps([{
        "id": "t1",
        "description": "print hello world",
        "acceptance_criteria": ["prints hello world"],
        "depends_on": [],
    }])


def _multi_subtask_response(prompt, *, model, system="", max_tokens=4096):
    return json.dumps([
        {
            "id": "A",
            "description": "implement add",
            "acceptance_criteria": ["adds two integers"],
            "depends_on": [],
        },
        {
            "id": "B",
            "description": "implement subtract",
            "acceptance_criteria": ["subtracts two integers"],
            "depends_on": ["A"],
        },
    ])


def _diamond_dag_response(prompt, *, model, system="", max_tokens=4096):
    return json.dumps([
        {
            "id": "A", "description": "base",
            "acceptance_criteria": ["a"], "depends_on": [],
        },
        {
            "id": "B", "description": "mid1",
            "acceptance_criteria": ["b"], "depends_on": ["A"],
        },
        {
            "id": "C", "description": "mid2",
            "acceptance_criteria": ["c"], "depends_on": ["A"],
        },
        {
            "id": "D", "description": "top",
            "acceptance_criteria": ["d"], "depends_on": ["B", "C"],
        },
    ])


def _cyclic_response(prompt, *, model, system="", max_tokens=4096):
    return json.dumps([
        {
            "id": "X", "description": "x",
            "acceptance_criteria": ["x"], "depends_on": ["Y"],
        },
        {
            "id": "Y", "description": "y",
            "acceptance_criteria": ["y"], "depends_on": ["X"],
        },
    ])


def _too_many_subtasks(prompt, *, model, system="", max_tokens=4096):
    return json.dumps([
        {
            "id": f"t{i}", "description": f"task {i}",
            "acceptance_criteria": [f"c{i}"], "depends_on": [],
        }
        for i in range(15)
    ])


@patch("builder_agent.plan.ask", side_effect=_single_subtask_response)
def test_plan_single_subtask(mock_ask):
    p = plan(SIMPLE_SPEC)
    assert len(p.subtasks) == 1
    assert p.subtasks[0].id == "t1"


@patch("builder_agent.plan.ask", side_effect=_multi_subtask_response)
def test_plan_ordered_subtasks(mock_ask):
    p = plan(SPEC)
    assert len(p.subtasks) == 2
    ids = [s.id for s in p.subtasks]
    assert ids.index("A") < ids.index("B")


@patch("builder_agent.plan.ask", side_effect=_diamond_dag_response)
def test_plan_topo_sort_diamond(mock_ask):
    p = plan(SPEC)
    ids = [s.id for s in p.subtasks]
    assert ids.index("A") < ids.index("B")
    assert ids.index("A") < ids.index("C")
    assert ids.index("B") < ids.index("D")
    assert ids.index("C") < ids.index("D")


@patch("builder_agent.plan.ask", side_effect=_cyclic_response)
def test_plan_rejects_cyclic(mock_ask):
    try:
        plan(SPEC)
        assert False, "Should have raised on cycle"
    except ValueError as e:
        assert "Cyclic" in str(e)


def test_topo_sort_direct():
    subtasks = [
        SubTask(id="A", description="a", acceptance_criteria=["a"]),
        SubTask(id="B", description="b", acceptance_criteria=["b"], depends_on=["A"]),
        SubTask(id="C", description="c", acceptance_criteria=["c"], depends_on=["A"]),
        SubTask(
            id="D", description="d",
            acceptance_criteria=["d"], depends_on=["B", "C"],
        ),
    ]
    ordered = _topo_sort(subtasks)
    ids = [s.id for s in ordered]
    assert ids.index("A") < ids.index("B")
    assert ids.index("A") < ids.index("C")
    assert ids.index("B") < ids.index("D")
    assert ids.index("C") < ids.index("D")


@patch("builder_agent.plan.ask", side_effect=_too_many_subtasks)
@patch("builder_agent.config.MAX_SUBTASKS", 8)
def test_plan_respects_max_subtasks(mock_ask):
    p = plan(SPEC)
    assert len(p.subtasks) <= 8


class _StubEmbedder:
    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 8
        for i, ch in enumerate(text):
            vec[i % 8] += ord(ch) / 1000.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


@patch("builder_agent.plan.ask", side_effect=_single_subtask_response)
def test_plan_injects_memory_records(mock_ask):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    emb = _StubEmbedder()
    mem = Memory(db_path=path, embedder=emb)

    mem.store(MemoryRecord(
        request="build a calculator",
        output_type="python_module",
        subtask_desc="A -> B plan",
        failures=[],
        fix_summary="final verify passed",
        final_code="",
        embedding=emb.embed("build a calculator"),
        record_type="plan",
    ))

    plan(SPEC, memory=mem)
    prompt_arg = mock_ask.call_args[0][0]
    assert "Prior plan decompositions" in prompt_arg
    assert "final verify passed" in prompt_arg

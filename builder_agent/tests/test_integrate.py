import ast

from builder_agent.integrate import integrate
from builder_agent.schemas import Plan, Spec, SubTask

SPEC = Spec(
    request="calculator",
    description="A CLI calculator",
    acceptance_criteria=["adds", "subtracts"],
    assumptions=[],
    output_type="python_module",
)

PLAN = Plan(subtasks=[
    SubTask(id="t1", description="add", acceptance_criteria=["adds"]),
    SubTask(
        id="t2", description="subtract",
        acceptance_criteria=["subtracts"], depends_on=["t1"],
    ),
])


def test_integrate_produces_valid_python():
    outputs = {
        "t1": "import math\n\ndef add(a, b):\n    return a + b",
        "t2": "import os\n\ndef subtract(a, b):\n    return a - b",
    }
    result = integrate(SPEC, outputs, PLAN)
    ast.parse(result)


def test_integrate_correct_order():
    outputs = {
        "t1": "def add(a, b):\n    return a + b",
        "t2": "def subtract(a, b):\n    return a - b",
    }
    result = integrate(SPEC, outputs, PLAN)
    assert result.index("add") < result.index("subtract")


def test_integrate_dedupes_imports():
    outputs = {
        "t1": "import math\nimport os\n\ndef add(a, b):\n    return a + b",
        "t2": "import math\nimport json\n\ndef subtract(a, b):\n    return a - b",
    }
    result = integrate(SPEC, outputs, PLAN)
    assert result.count("import math") == 1
    assert "import os" in result
    assert "import json" in result


def test_integrate_has_all():
    outputs = {
        "t1": "def add(a, b):\n    return a + b",
        "t2": "def subtract(a, b):\n    return a - b",
    }
    result = integrate(SPEC, outputs, PLAN)
    assert "__all__" in result
    assert "add" in result
    assert "subtract" in result


def test_integrate_syntax_check():
    outputs = {
        "t1": "def add(a, b):\n    return a + b",
        "t2": "def subtract(a, b):\n    return a - b",
    }
    result = integrate(SPEC, outputs, PLAN)
    tree = ast.parse(result)
    assert tree is not None

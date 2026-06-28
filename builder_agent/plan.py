"""Decomposition of specification details into topological plans."""

from __future__ import annotations

import json

from builder_agent import config
from builder_agent.llm import ask, extract_json
from builder_agent.memory import Memory
from builder_agent.schemas import Plan, Spec, SubTask

_SYSTEM = (
    "You are a task planner. Decompose a spec into ordered subtasks. "
    "Each subtask has: id (str), description (str), "
    "acceptance_criteria (list[str]), depends_on (list[str] of prerequisite ids). "
    "Respond with ONLY a JSON array of subtask objects, no markdown fencing.\n\n"
    "CRITICAL RULES:\n"
    "- SIMPLE TASKS GET EXACTLY 1 SUBTASK. If the spec describes a single "
    "function, class, or small module — use 1 subtask. Do NOT split a simple "
    "function into 'implement', 'test', 'validate' subtasks.\n"
    "- Only split into multiple subtasks when the spec has genuinely "
    "independent components (e.g. a CLI + a library + a config parser).\n"
    "- Never create subtasks for testing, validation, or error handling — "
    "those are part of the main subtask's acceptance criteria.\n"
    "- Every acceptance criterion from the spec MUST be covered by at least "
    "one subtask's criteria — don't invent unrelated criteria.\n"
    "- depends_on must form a DAG (no cycles).\n"
    "- Max {max_subtasks} subtasks."
)

_PROMPT = (
    "Spec:\n"
    "  Description: {description}\n"
    "  Acceptance criteria:\n{criteria}\n"
    "  Output type: {output_type}\n\n"
    "{memory_block}"
    "How many subtasks? If this is a single function/class/module, "
    "use EXACTLY 1 subtask. Only use multiple if there are truly "
    "independent components. Max {max_subtasks}."
)


def _topo_sort(subtasks: list[SubTask]) -> list[SubTask]:
    by_id = {s.id: s for s in subtasks}
    visited: set[str] = set()
    order: list[SubTask] = []
    visiting: set[str] = set()

    def visit(sid: str) -> None:
        if sid in visiting:
            raise ValueError(f"Cyclic dependency detected involving '{sid}'")
        if sid in visited:
            return
        visiting.add(sid)
        for dep in by_id[sid].depends_on:
            if dep in by_id:
                visit(dep)
        visiting.remove(sid)
        visited.add(sid)
        order.append(by_id[sid])

    for s in subtasks:
        visit(s.id)
    return order


def plan(spec: Spec, memory: Memory | None = None) -> Plan:
    """Decompose a Spec object into a topologically sorted build Plan.

    Args:
        spec: Target build specifications structure.
        memory: Stored experiences database.

    Returns:
        The generated execution Plan containing subtasks.
    """
    criteria = "\n".join(f"  - {c}" for c in spec.acceptance_criteria)

    memory_block = ""
    if memory is not None:
        records = memory.retrieve(
            spec.request, k=config.MEMORY_TOP_K, record_type="plan"
        )
        if records:
            parts = []
            for r in records:
                parts.append(
                    f"- Prior plan for: {r.subtask_desc}\n"
                    f"  Outcome: {r.fix_summary}"
                )
            memory_block = (
                "Prior plan decompositions:\n"
                + "\n".join(parts) + "\n\n"
            )

    prompt = _PROMPT.format(
        description=spec.description,
        criteria=criteria,
        output_type=spec.output_type,
        memory_block=memory_block,
        max_subtasks=config.MAX_SUBTASKS,
    )
    system = _SYSTEM.format(max_subtasks=config.MAX_SUBTASKS)
    raw = ask(prompt, model=config.PLANNER_MODEL, system=system)
    data = json.loads(extract_json(raw))

    soft_cap = max(3, len(spec.acceptance_criteria) + 1)
    hard_cap = min(config.MAX_SUBTASKS, soft_cap)
    if len(data) > hard_cap:
        data = data[:hard_cap]

    subtasks = []
    for item in data:
        deps = item.get("depends_on", [])
        subtasks.append(
            SubTask(
                id=item["id"],
                description=item["description"],
                acceptance_criteria=item["acceptance_criteria"],
                depends_on=deps,
            )
        )

    ordered = _topo_sort(subtasks)
    return Plan(subtasks=ordered)

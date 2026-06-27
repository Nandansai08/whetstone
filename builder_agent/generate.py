from __future__ import annotations

from typing import Callable

from builder_agent import config
from builder_agent.config import ModelConfig
from builder_agent.llm import ask, ask_stream, strip_fences
from builder_agent.schemas import MemoryRecord, Spec, SubTask

_GENERATE_SYSTEM = (
    "You are an expert programmer. Produce ONLY the code — no markdown fencing, "
    "no explanations. The code must satisfy the acceptance criteria."
)

_GENERATE_PROMPT = (
    "Spec: {spec_desc}\n\n"
    "SubTask: {subtask_desc}\n"
    "Acceptance criteria:\n{criteria}\n\n"
    "{feedback_block}"
    "{hints_block}"
    "Write the code."
)

_CRITIQUE_SYSTEM = (
    "You are a code reviewer. Given code and its acceptance criteria, "
    "return an improved version. Output ONLY code, no commentary."
)

_CRITIQUE_PROMPT = (
    "Acceptance criteria:\n{criteria}\n\n"
    "Code to review and improve:\n{code}"
)


def generate(
    subtask: SubTask,
    spec: Spec,
    feedback: str | None = None,
    memory_hints: list[MemoryRecord] | None = None,
    worker_model: ModelConfig | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    criteria = "\n".join(f"- {c}" for c in subtask.acceptance_criteria)
    feedback_block = ""
    if feedback:
        feedback_block = (
            f"Previous attempt failed. Issues:\n{feedback}\n\n"
            "Fix these issues in your new attempt.\n\n"
        )
    hints_block = ""
    if memory_hints:
        hints_parts = []
        for h in memory_hints:
            hints_parts.append(
                f"- Similar task: {h.subtask_desc}\n"
                f"  Fix: {h.fix_summary}"
            )
        hints_block = (
            "Hints from similar past builds:\n"
            + "\n".join(hints_parts) + "\n\n"
        )

    prompt = _GENERATE_PROMPT.format(
        spec_desc=spec.description,
        subtask_desc=subtask.description,
        criteria=criteria,
        feedback_block=feedback_block,
        hints_block=hints_block,
    )
    model = worker_model or config.WORKER_MODEL
    if on_chunk is None:
        return strip_fences(
            ask(prompt, model=model, system=_GENERATE_SYSTEM)
        )

    chunks = []
    for chunk in ask_stream(prompt, model=model, system=_GENERATE_SYSTEM):
        chunks.append(chunk)
        on_chunk(chunk)
    return strip_fences("".join(chunks))


def self_critique(
    code: str,
    subtask: SubTask,
    worker_model: ModelConfig | None = None,
) -> str:
    criteria = "\n".join(f"- {c}" for c in subtask.acceptance_criteria)
    prompt = _CRITIQUE_PROMPT.format(criteria=criteria, code=code)
    model = worker_model or config.WORKER_MODEL
    return strip_fences(
        ask(prompt, model=model, system=_CRITIQUE_SYSTEM)
    )

"""Code generation and self-critique review worker routines."""

from __future__ import annotations

from typing import Callable

from builder_agent import config
from builder_agent.config import ModelConfig
from builder_agent.llm import ask, ask_stream, strip_fences
from builder_agent.schemas import MemoryRecord, Spec, SubTask

_GENERATE_SYSTEM_TEMPLATE = (
    "You are an expert {language} programmer. Produce ONLY the code — "
    "no markdown fencing, no explanations. "
    "The code must satisfy the acceptance criteria."
)

_GENERATE_PROMPT = (
    "Spec: {spec_desc}\n\n"
    "SubTask: {subtask_desc}\n"
    "Acceptance criteria:\n{criteria}\n\n"
    "{feedback_block}"
    "{hints_block}"
    "Write the {language} code."
)

_CRITIQUE_SYSTEM_TEMPLATE = (
    "You are a {language} code reviewer. Given code and its acceptance criteria, "
    "return an improved version. Output ONLY {language} code, no commentary."
)

_CRITIQUE_PROMPT = (
    "Acceptance criteria:\n{criteria}\n\n"
    "Code to review and improve:\n{code}"
)

_LANG_CONFIGS = {
    "python": "Python",
    "python_module": "Python",
    "python_package": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
}


def _get_lang_name(output_type: str) -> str:
    return _LANG_CONFIGS.get(output_type, "Python")


def generate(
    subtask: SubTask,
    spec: Spec,
    feedback: str | None = None,
    memory_hints: list[MemoryRecord] | None = None,
    worker_model: ModelConfig | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> str:
    """Generate source code for a specific plan subtask using worker models.

    Args:
        subtask: The subtask defining requirements and acceptance criteria.
        spec: The global specification structure.
        feedback: Reviewer/verifier issues from any previous failed runs.
        memory_hints: Stored memory guides representing past build lessons.
        worker_model: Specific model configuration profile override.
        on_chunk: Callback callback stream to stream token chunks during query.

    Returns:
        The generated raw source code block.
    """
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

    lang_name = _get_lang_name(spec.output_type)

    prompt = _GENERATE_PROMPT.format(
        spec_desc=spec.description,
        subtask_desc=subtask.description,
        criteria=criteria,
        feedback_block=feedback_block,
        hints_block=hints_block,
        language=lang_name,
    )
    system = _GENERATE_SYSTEM_TEMPLATE.format(language=lang_name)
    model = worker_model or config.WORKER_MODEL
    if on_chunk is None:
        return strip_fences(
            ask(prompt, model=model, system=system)
        )

    chunks = []
    for chunk in ask_stream(prompt, model=model, system=system):
        chunks.append(chunk)
        on_chunk(chunk)
    return strip_fences("".join(chunks))


def self_critique(
    code: str,
    subtask: SubTask,
    worker_model: ModelConfig | None = None,
    output_type: str = "python",
) -> str:
    """Analyze and refine generated code drafts before passing to the verifier.

    Args:
        code: Draft version of source code to analyze.
        subtask: SubTask definition context representing goals.
        worker_model: Specific model configuration profile override.

    Returns:
        The refined/improved version of the source code.
    """
    criteria = "\n".join(f"- {c}" for c in subtask.acceptance_criteria)
    prompt = _CRITIQUE_PROMPT.format(criteria=criteria, code=code)
    model = worker_model or config.WORKER_MODEL
    lang_name = _get_lang_name(output_type)
    system = _CRITIQUE_SYSTEM_TEMPLATE.format(language=lang_name)
    return strip_fences(
        ask(prompt, model=model, system=system)
    )

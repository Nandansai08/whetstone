"""Requirements clarification and specification extraction module."""

from __future__ import annotations

import json

from builder_agent import config
from builder_agent.llm import ask, extract_json
from builder_agent.schemas import Spec

_SYSTEM = (
    "You are a requirements analyst. Given a user request, produce a JSON object "
    "with keys: description (str), acceptance_criteria (list[str]), "
    "assumptions (list[str]), output_type (str). "
    "acceptance_criteria must be objective, checkable statements. "
    "output_type is one of: python, python_module, python_package, "
    "javascript, typescript, sql, pipeline."
)

_CLARIFY_PROMPT = (
    "User request: {request}\n\n"
    "Ask up to 3 high-value clarifying questions, then produce the spec. "
    "If answers are provided below, use them instead of asking.\n"
    "{answers_block}"
    "Respond with ONLY the JSON object, no markdown fencing."
)

_AMBIGUITY_SYSTEM = (
    "You are a requirements analyst. Inspect the user's request for software "
    "generation. If the request is ambiguous, lacks concrete specifications, "
    "or requires decisions (e.g., input/output format, edge case handling, "
    "libraries to use), generate 1 to 3 high-value, specific clarifying questions. "
    "If the request is already clear and detailed enough to build without "
    "additional clarification, return an empty JSON list []. "
    "Return ONLY a valid JSON list of strings, with no explanation or markdown fencing."
)

_AMBIGUITY_PROMPT = (
    "User request: {request}\n\n"
    "Generate clarifying questions if needed. Return ONLY a JSON list of strings."
)


def detect_ambiguity(request: str) -> list[str]:
    """Analyze the request for ambiguity and return up to 3 clarifying questions.

    If the request is clear or if an error occurs, returns an empty list.
    """
    raw = ask(
        _AMBIGUITY_PROMPT.format(request=request),
        model=config.PLANNER_MODEL,
        system=_AMBIGUITY_SYSTEM,
    )
    try:
        data = json.loads(extract_json(raw))
        if isinstance(data, list):
            questions = [str(q).strip() for q in data if q]
            return questions[:3]
    except Exception:
        pass
    return []



def clarify(request: str, *, interactive: bool = True) -> Spec:
    """Clarify the user request and extract a concrete Specification object.

    Args:
        request: The raw build request string.
        interactive: If True, prompt the user for clarification.

    Returns:
        A Spec object containing the validated description, acceptance criteria,
        assumptions, and output type.
    """
    answers_block = ""
    if not interactive:
        answers_block = "No interactive session — use sensible defaults.\n"

    prompt = _CLARIFY_PROMPT.format(
        request=request, answers_block=answers_block
    )
    raw = ask(prompt, model=config.PLANNER_MODEL, system=_SYSTEM)
    data = json.loads(extract_json(raw))

    return Spec(
        request=request,
        description=data["description"],
        acceptance_criteria=data["acceptance_criteria"],
        assumptions=data.get("assumptions", []),
        output_type=data.get("output_type", "python_module"),
    )

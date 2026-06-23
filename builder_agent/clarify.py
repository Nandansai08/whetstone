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
    "output_type is one of: python_module, sql, pipeline."
)

_CLARIFY_PROMPT = (
    "User request: {request}\n\n"
    "Ask up to 3 high-value clarifying questions, then produce the spec. "
    "If answers are provided below, use them instead of asking.\n"
    "{answers_block}"
    "Respond with ONLY the JSON object, no markdown fencing."
)


def clarify(request: str, *, interactive: bool = True) -> Spec:
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

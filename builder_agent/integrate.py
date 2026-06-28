from __future__ import annotations

import ast
import json
import re

from builder_agent import config
from builder_agent.llm import ask, extract_json
from builder_agent.safety import is_safe_relative_path
from builder_agent.schemas import Plan, Spec

_INTEGRATE_PACKAGE_SYSTEM = (
    "You are an expert software architect. You are given a specifications list "
    "and a collection of Python code blocks generated for subtasks. "
    "Structure these code blocks into a cohesive, valid, multi-file Python package. "
    "Return ONLY a JSON object mapping file paths (relative to the package root) "
    "to their file contents. Do not output markdown code fences, "
    "do not output explanations."
)

_INTEGRATE_PACKAGE_PROMPT = (
    "Spec Description: {spec_desc}\n\n"
    "Subtask code outputs:\n{code_outputs}\n\n"
    "Generate the Python package structure. Ensure all files use valid "
    "absolute/relative imports to refer to other files in the package. "
    "Create a proper __init__.py that exposes the main public interface. "
    "Return a JSON object: {{\"path\": \"content\"}}."
)


def _extract_imports(code: str) -> tuple[list[str], str]:
    import_lines: list[str] = []
    other_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            import_lines.append(stripped)
        else:
            other_lines.append(line)
    return import_lines, "\n".join(other_lines)


def _extract_public_names(code: str) -> list[str]:
    names: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return names
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append(target.id)
    return names


def integrate(spec: Spec, outputs: dict[str, str], plan: Plan) -> str | dict[str, str]:
    """Integrate outputs into a single module or a directory file tree (package)."""
    if spec.output_type == "python_package":
        code_outputs = ""
        for subtask in plan.subtasks:
            code = outputs.get(subtask.id, "")
            if code:
                code_outputs += (
                    f"### Subtask {subtask.id} ({subtask.description}):\n"
                    f"{code}\n\n"
                )

        prompt = _INTEGRATE_PACKAGE_PROMPT.format(
            spec_desc=spec.description,
            code_outputs=code_outputs.strip(),
        )
        raw = ask(
            prompt,
            model=config.PLANNER_MODEL,
            system=_INTEGRATE_PACKAGE_SYSTEM,
        )
        package_data = json.loads(extract_json(raw))

        if not isinstance(package_data, dict):
            raise ValueError(
                f"Integrator LLM did not return a dictionary mapping: {raw}"
            )

        for path in package_data.keys():
            if not is_safe_relative_path(path):
                raise ValueError(
                    f"Unsafe path traversal/absolute path detected: {path}"
                )

        return {str(k): str(v) for k, v in package_data.items()}

    # Existing single module logic
    all_imports: list[str] = []
    all_bodies: list[str] = []

    for subtask in plan.subtasks:
        code = outputs.get(subtask.id, "")
        if not code:
            continue
        imports, body = _extract_imports(code)
        all_imports.extend(imports)
        all_bodies.append(body.strip())

    seen: set[str] = set()
    deduped_imports: list[str] = []
    for imp in all_imports:
        normalized = re.sub(r"\s+", " ", imp).strip()
        if normalized not in seen:
            seen.add(normalized)
            deduped_imports.append(imp)

    parts = []
    if deduped_imports:
        parts.append("\n".join(deduped_imports))
        parts.append("")
    parts.append("\n\n\n".join(all_bodies))

    combined = "\n".join(parts)

    public_names = _extract_public_names(combined)
    if public_names:
        all_line = "__all__ = " + repr(public_names)
        combined = combined + "\n\n\n" + all_line + "\n"

    ast.parse(combined)

    return combined

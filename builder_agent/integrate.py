from __future__ import annotations

import ast
import re

from builder_agent.schemas import Plan, Spec


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


def integrate(spec: Spec, outputs: dict[str, str], plan: Plan) -> str:
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

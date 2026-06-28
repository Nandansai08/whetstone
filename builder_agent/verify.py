from __future__ import annotations

import json
import sqlite3
from typing import Protocol

from builder_agent import config
from builder_agent.llm import ask, extract_json, strip_fences
from builder_agent.sandbox import run_code
from builder_agent.schemas import SubTask, Verdict

_TEST_SYSTEM = (
    "You are a test engineer. Given acceptance criteria and code, "
    "write pytest-style tests that verify each criterion. "
    "Output ONLY executable Python test code, no markdown fencing."
)

_TEST_PROMPT = (
    "Acceptance criteria:\n{criteria}\n\n"
    "Code under test:\n{code}\n\n"
    "Write tests that import nothing external — inline the code if needed."
)

_JUDGE_SYSTEM = (
    "You are a code judge. Score the code 0-10 against the criteria rubric. "
    "Respond with ONLY a JSON object: "
    '{"score": <int>, "issues": [<str>, ...]}'
)

_JUDGE_PROMPT = (
    "Acceptance criteria:\n{criteria}\n\n"
    "Code:\n{code}\n\n"
    "Execution output:\n{exec_output}\n\n"
    "Score 0-10. List concrete issues."
)


def make_tests(subtask: SubTask, code: str) -> str:
    criteria = "\n".join(f"- {c}" for c in subtask.acceptance_criteria)
    prompt = _TEST_PROMPT.format(criteria=criteria, code=code)
    return ask(prompt, model=config.WORKER_MODEL, system=_TEST_SYSTEM)


class Verifier(Protocol):
    def verify(self, subtask: SubTask, code: str | dict[str, str]) -> Verdict:
        ...


class GenericVerifier:
    def verify(self, subtask: SubTask, code: str) -> Verdict:
        test_code = strip_fences(make_tests(subtask, code))
        full_code = code + "\n\n" + test_code
        tests_passed, exec_output = run_code(
            full_code, timeout=config.EXEC_TIMEOUT
        )

        if not tests_passed:
            return Verdict(
                passed=False,
                score=0,
                tests_passed=False,
                issues=[f"Tests failed: {exec_output}"],
                exec_output=exec_output,
            )

        criteria = "\n".join(f"- {c}" for c in subtask.acceptance_criteria)
        prompt = _JUDGE_PROMPT.format(
            criteria=criteria, code=code, exec_output=exec_output
        )
        raw = ask(prompt, model=config.JUDGE_MODEL, system=_JUDGE_SYSTEM)
        data = json.loads(extract_json(raw))
        score = int(data["score"])
        issues = data.get("issues", [])

        return Verdict(
            passed=score >= config.SCORE_THRESHOLD,
            score=score,
            tests_passed=True,
            issues=issues,
            exec_output=exec_output,
        )


class PythonPackageVerifier:
    def verify(self, subtask: SubTask, code: dict[str, str]) -> Verdict:
        # Format package files for prompt context
        code_str_for_prompt = ""
        for path, content in code.items():
            code_str_for_prompt += f"# File: {path}\n{content}\n\n"

        test_code = strip_fences(make_tests(subtask, code_str_for_prompt))

        # Build a self-extracting, isolated test execution script
        full_code = f"""import os
import sys
import tempfile
import shutil

test_dir = tempfile.mkdtemp()
orig_dir = os.getcwd()
os.chdir(test_dir)
sys.path.insert(0, test_dir)

try:
    files = {repr(code)}
    for path, content in files.items():
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    with open("test_code.py", "w", encoding="utf-8") as f:
        f.write({repr(test_code)})

    try:
        import pytest
        ret = pytest.main(["test_code.py", "-v"])
        sys.exit(ret)
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_code", "test_code.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        failed = False
        for name in dir(module):
            if name.startswith("test_") and callable(getattr(module, name)):
                try:
                    getattr(module, name)()
                except AssertionError as e:
                    print(f"Test {{name}} failed: {{e}}", file=sys.stderr)
                    failed = True
                except Exception as e:
                    print(f"Test {{name}} errored: {{e}}", file=sys.stderr)
                    failed = True
        sys.exit(1 if failed else 0)
finally:
    os.chdir(orig_dir)
    try:
        shutil.rmtree(test_dir)
    except Exception:
        pass
"""

        tests_passed, exec_output = run_code(
            full_code, timeout=config.EXEC_TIMEOUT
        )

        if not tests_passed:
            return Verdict(
                passed=False,
                score=0,
                tests_passed=False,
                issues=[f"Tests failed: {exec_output}"],
                exec_output=exec_output,
            )

        criteria = "\n".join(f"- {c}" for c in subtask.acceptance_criteria)
        prompt = _JUDGE_PROMPT.format(
            criteria=criteria, code=code_str_for_prompt, exec_output=exec_output
        )
        raw = ask(prompt, model=config.JUDGE_MODEL, system=_JUDGE_SYSTEM)
        data = json.loads(extract_json(raw))
        score = int(data["score"])
        issues = data.get("issues", [])

        return Verdict(
            passed=score >= config.SCORE_THRESHOLD,
            score=score,
            tests_passed=True,
            issues=issues,
            exec_output=exec_output,
        )


_SQL_JUDGE_SYSTEM = (
    "You are a SQL code judge. Score the SQL code 0-10 against the criteria rubric. "
    "Respond with ONLY a JSON object: "
    '{"score": <int>, "issues": [<str>, ...]}'
)

_SQL_JUDGE_PROMPT = (
    "Acceptance criteria:\n{criteria}\n\n"
    "SQL Code:\n{code}\n\n"
    "Database Schema / Execution Status:\n{schema_or_error}\n\n"
    "Score 0-10. List concrete issues. If there was a SQL execution error "
    "due to standard SQLite lacking support for another SQL dialect (e.g. "
    "Postgres CTEs, JSON, spatial, or dialect-specific datatypes like SERIAL), "
    "do not penalize the score if the SQL code is otherwise correct for its "
    "target dialect."
)


_SAFE_SQL_OP_NAMES = [
    "SQLITE_SELECT",
    "SQLITE_INSERT",
    "SQLITE_UPDATE",
    "SQLITE_DELETE",
    "SQLITE_CREATE_TABLE",
    "SQLITE_CREATE_TEMP_TABLE",
    "SQLITE_CREATE_VIEW",
    "SQLITE_CREATE_TEMP_VIEW",
    "SQLITE_CREATE_INDEX",
    "SQLITE_CREATE_TEMP_INDEX",
    "SQLITE_DROP_TABLE",
    "SQLITE_DROP_TEMP_TABLE",
    "SQLITE_DROP_VIEW",
    "SQLITE_DROP_TEMP_VIEW",
    "SQLITE_DROP_INDEX",
    "SQLITE_DROP_TEMP_INDEX",
    "SQLITE_TRANSACTION",
    "SQLITE_READ",
    "SQLITE_FUNCTION",
    "SQLITE_ALTER_TABLE",
    "SQLITE_SAVEPOINT",
    "SQLITE_PRAGMA",
    "SQLITE_RECURSIVE",
]

_SAFE_SQL_OPS = {
    getattr(sqlite3, name)
    for name in _SAFE_SQL_OP_NAMES
    if hasattr(sqlite3, name)
}


def _sql_authorizer(action, arg1, arg2, dbname, source_principal):
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY
    if action in _SAFE_SQL_OPS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SqlVerifier:
    def verify(self, subtask: SubTask, code: str) -> Verdict:
        schema_or_error = ""
        try:
            conn = sqlite3.connect(":memory:")
            conn.set_authorizer(_sql_authorizer)
            cursor = conn.cursor()
            cursor.executescript(code)

            # Extract schema details if applicable
            cursor.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table'"
            )
            tables = cursor.fetchall()
            if tables:
                schema_or_error = "Generated Schema:\n" + "\n".join(
                    f"Table: {name}\nSQL: {sql}" for name, sql in tables
                )
            else:
                schema_or_error = "No tables created."
            conn.close()
        except Exception as e:
            schema_or_error = f"SQLite execution failed with error: {e}"

        criteria = "\n".join(f"- {c}" for c in subtask.acceptance_criteria)
        prompt = _SQL_JUDGE_PROMPT.format(
            criteria=criteria, code=code, schema_or_error=schema_or_error
        )
        try:
            raw = ask(prompt, model=config.JUDGE_MODEL, system=_SQL_JUDGE_SYSTEM)
            data = json.loads(extract_json(raw))
            score = int(data["score"])
            issues = data.get("issues", [])
        except Exception as e:
            score = 0
            issues = [f"SQL Judge evaluation failed: {e}"]

        return Verdict(
            passed=score >= config.SCORE_THRESHOLD,
            score=score,
            tests_passed=True,
            issues=issues,
            exec_output=schema_or_error,
        )


_VERIFIERS: dict[str, Verifier] = {
    "sql": SqlVerifier(),
    "python_module": GenericVerifier(),
    "python_package": PythonPackageVerifier(),
}


def verify(
    subtask: SubTask, code: str | dict[str, str], output_type: str = "python_module"
) -> Verdict:
    verifier = _VERIFIERS.get(output_type, _VERIFIERS["python_module"])
    return verifier.verify(subtask, code)

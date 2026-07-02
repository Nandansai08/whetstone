import os
import subprocess
import tempfile
from typing import Any

from builder_agent.plugin_system.base import PluginContext, PluginVerificationResult


class LinterPlugin:
    """Built-in verifier plugin running Ruff for Python and ESLint for JS/TS."""

    def verify(
        self, subtask: Any, code: str | dict[str, str], context: PluginContext
    ) -> PluginVerificationResult | None:
        """Run Ruff (Python) or ESLint (JS/TS) on the generated code block."""
        output_type = context.output_type
        if output_type in ("python", "python_module", "python_package"):
            return self._lint_python(code)
        elif output_type in ("javascript", "typescript"):
            return self._lint_js_ts(code, output_type)

        # Skip unsupported output types cleanly
        return PluginVerificationResult(
            passed=True,
            issues=[],
            exec_output=f"Skipping linter for unsupported output type: {output_type}",
            blocking=False,
        )

    def _lint_python(self, code: str | dict[str, str]) -> PluginVerificationResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            if isinstance(code, dict):
                for path, content in code.items():
                    full_path = os.path.join(tmpdir, path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(content)
                target = tmpdir
            else:
                target = os.path.join(tmpdir, "code.py")
                with open(target, "w", encoding="utf-8") as f:
                    f.write(code)

            try:
                res = subprocess.run(
                    ["ruff", "check", target],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                passed = res.returncode == 0
                issues = []
                if not passed:
                    issues = [
                        line.strip()
                        for line in res.stdout.splitlines()
                        if line.strip()
                    ]
                    if not issues:
                        issues = [res.stderr.strip()]
                return PluginVerificationResult(
                    passed=passed,
                    issues=issues,
                    exec_output=res.stdout + res.stderr,
                    blocking=True,
                )
            except FileNotFoundError:
                return PluginVerificationResult(
                    passed=True,
                    issues=[],
                    exec_output="Ruff executable not found. Skipped python linting.",
                    blocking=False,
                )

    def _lint_js_ts(
        self, code: str | dict[str, str], output_type: str
    ) -> PluginVerificationResult:
        if isinstance(code, dict):
            return PluginVerificationResult(
                passed=True,
                issues=[],
                exec_output="ESLint multi-file package linting is not supported.",
                blocking=False,
            )

        ext = ".ts" if output_type == "typescript" else ".js"
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, f"code{ext}")
            with open(target, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                cmd = ["npx", "eslint", target]
                if os.name == "nt":
                    res = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                        shell=True,
                    )
                else:
                    res = subprocess.run(
                        cmd, capture_output=True, text=True, check=False
                    )
                passed = res.returncode == 0
                issues = []
                if not passed:
                    issues = [
                        line.strip()
                        for line in res.stdout.splitlines()
                        if line.strip()
                    ]
                    if not issues:
                        issues = [res.stderr.strip()]
                return PluginVerificationResult(
                    passed=passed,
                    issues=issues,
                    exec_output=res.stdout + res.stderr,
                    blocking=True,
                )
            except FileNotFoundError:
                return PluginVerificationResult(
                    passed=True,
                    issues=[],
                    exec_output="ESLint via npx not found. Skipped JS/TS linting.",
                    blocking=False,
                )

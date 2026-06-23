from __future__ import annotations

import subprocess
import sys
import tempfile


def run_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import uuid

from builder_agent import config


def run_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute Python code in a sandboxed environment."""
    if config.SANDBOX_BACKEND == "container":
        return _run_in_container(code, timeout)
    return _run_in_subprocess(code, timeout)


def _run_in_subprocess(code: str, timeout: int) -> tuple[bool, str]:
    """Execute Python code locally in a subprocess."""
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


def _run_in_container(code: str, timeout: int) -> tuple[bool, str]:
    """Execute Python code securely inside a Docker or Podman container."""
    engine = config.SANDBOX_ENGINE
    if not shutil.which(engine):
        raise RuntimeError(
            f"Sandbox engine '{engine}' is not available. "
            f"Please verify it is installed and on the system PATH."
        )

    container_name = f"whetstone-sandbox-{uuid.uuid4().hex[:12]}"
    cmd = [engine, "run", "--name", container_name, "--rm", "-i"]

    if not config.SANDBOX_NETWORK_ACCESS:
        cmd.extend(["--network", "none"])
    if config.SANDBOX_MEMORY_LIMIT:
        cmd.extend(["-m", str(config.SANDBOX_MEMORY_LIMIT)])
    if config.SANDBOX_CPU_LIMIT:
        cmd.extend(["--cpus", str(config.SANDBOX_CPU_LIMIT)])

    # Standard security hardening flags
    cmd.extend([
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "32",
        "--user", "65534:65534",
        config.SANDBOX_IMAGE,
        "python",
    ])

    try:
        result = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check for container engine/daemon setup issues
        daemon_errors = [
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "connecting to the service failed",
            "cannot connect to the podman socket",
            "podman service",
            "daemon unreachable",
        ]
        stderr_lower = result.stderr.lower()
        is_daemon_error = any(msg in stderr_lower for msg in daemon_errors)

        if result.returncode == 125 or is_daemon_error:
            raise RuntimeError(
                f"Sandbox engine '{engine}' failed to execute. "
                f"Please verify that the daemon is running. "
                f"Details: {result.stderr.strip()}"
            )

        output = result.stdout + result.stderr
        # Returncode 137 typically indicates container was OOM killed
        if result.returncode == 137:
            return False, "Execution failed: Memory limit exceeded (OOM)"
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        subprocess.run([engine, "rm", "-f", container_name], capture_output=True)
        return False, f"Timeout after {timeout}s"
    except KeyboardInterrupt:
        subprocess.run([engine, "rm", "-f", container_name], capture_output=True)
        raise
    except OSError as e:
        subprocess.run([engine, "rm", "-f", container_name], capture_output=True)
        raise RuntimeError(f"Failed to execute container: {e}") from e

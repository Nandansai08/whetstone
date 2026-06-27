import os
import shutil
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from builder_agent import config
from builder_agent.sandbox import run_code


def test_good_code_passes():
    passed, output = run_code("print('hello')")
    assert passed is True
    assert "hello" in output


def test_bad_code_fails():
    passed, output = run_code("raise ValueError('boom')")
    assert passed is False
    assert "ValueError" in output


def test_syntax_error_fails():
    passed, output = run_code("def f(\n")
    assert passed is False


def test_timeout_kills_infinite_loop():
    passed, output = run_code("while True: pass", timeout=2)
    assert passed is False
    assert "Timeout" in output


def test_exit_code_nonzero_fails():
    passed, output = run_code("import sys; sys.exit(1)")
    assert passed is False


def test_multiline_output():
    code = "print('line1')\nprint('line2')"
    passed, output = run_code(code)
    assert passed is True
    assert "line1" in output
    assert "line2" in output


def test_container_command_construction_docker(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "container")
    monkeypatch.setattr(config, "SANDBOX_ENGINE", "docker")
    monkeypatch.setattr(config, "SANDBOX_IMAGE", "python:3.11-slim")
    monkeypatch.setattr(config, "SANDBOX_MEMORY_LIMIT", "256m")
    monkeypatch.setattr(config, "SANDBOX_CPU_LIMIT", 1.0)
    monkeypatch.setattr(config, "SANDBOX_NETWORK_ACCESS", False)

    monkeypatch.setattr(
        shutil,
        "which",
        lambda x: "/usr/bin/docker" if x == "docker" else None,
    )

    called_cmds = []

    def mock_run(cmd, *args, **kwargs):
        called_cmds.append(cmd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "ok\n"
        mock_res.stderr = ""
        return mock_res

    monkeypatch.setattr(subprocess, "run", mock_run)

    passed, output = run_code("print('hello')")
    assert passed is True

    # No pre-check ps, only the main run command
    assert len(called_cmds) == 1
    run_cmd = called_cmds[0]
    assert run_cmd[0] == "docker"
    assert "run" in run_cmd
    assert "--name" in run_cmd
    idx = run_cmd.index("--name")
    assert run_cmd[idx + 1].startswith("whetstone-sandbox-")
    assert "--rm" in run_cmd
    assert "-i" in run_cmd
    assert "--network" in run_cmd
    assert "none" in run_cmd
    assert "-m" in run_cmd
    assert "256m" in run_cmd
    assert "--cpus" in run_cmd
    assert "1.0" in run_cmd
    assert "--read-only" in run_cmd
    assert "--tmpfs" in run_cmd
    assert "/tmp:rw,noexec,nosuid,size=64m" in run_cmd
    assert "--cap-drop" in run_cmd
    assert "ALL" in run_cmd
    assert "--security-opt" in run_cmd
    assert "no-new-privileges" in run_cmd
    assert "--pids-limit" in run_cmd
    assert "32" in run_cmd
    assert "--user" in run_cmd
    assert "65534:65534" in run_cmd
    assert "python:3.11-slim" in run_cmd
    assert "python" in run_cmd


def test_container_command_construction_podman(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "container")
    monkeypatch.setattr(config, "SANDBOX_ENGINE", "podman")
    monkeypatch.setattr(config, "SANDBOX_IMAGE", "python:3.11-slim")
    monkeypatch.setattr(config, "SANDBOX_MEMORY_LIMIT", "512m")
    monkeypatch.setattr(config, "SANDBOX_CPU_LIMIT", 0.5)
    monkeypatch.setattr(config, "SANDBOX_NETWORK_ACCESS", True)

    monkeypatch.setattr(
        shutil,
        "which",
        lambda x: "/usr/bin/podman" if x == "podman" else None,
    )

    called_cmds = []

    def mock_run(cmd, *args, **kwargs):
        called_cmds.append(cmd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "ok\n"
        mock_res.stderr = ""
        return mock_res

    monkeypatch.setattr(subprocess, "run", mock_run)

    passed, output = run_code("print('hello')")
    assert passed is True

    assert len(called_cmds) == 1
    run_cmd = called_cmds[0]
    assert run_cmd[0] == "podman"
    assert "run" in run_cmd
    # Should not have network none because network access is True
    assert "--network" not in run_cmd
    assert "-m" in run_cmd
    assert "512m" in run_cmd
    assert "--cpus" in run_cmd
    assert "0.5" in run_cmd


def test_container_engine_missing_raises(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "container")
    monkeypatch.setattr(config, "SANDBOX_ENGINE", "missing-engine")
    monkeypatch.setattr(shutil, "which", lambda x: None)

    with pytest.raises(RuntimeError) as exc_info:
        run_code("print('hello')")
    assert "missing-engine" in str(exc_info.value)
    assert "not available" in str(exc_info.value)


def test_container_daemon_unreachable_raises(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "container")
    monkeypatch.setattr(config, "SANDBOX_ENGINE", "docker")
    monkeypatch.setattr(
        shutil,
        "which",
        lambda x: "/usr/bin/docker" if x == "docker" else None,
    )

    def mock_run(cmd, *args, **kwargs):
        mock_res = MagicMock()
        mock_res.returncode = 125
        mock_res.stdout = ""
        mock_res.stderr = "Cannot connect to the Docker daemon"
        return mock_res

    monkeypatch.setattr(subprocess, "run", mock_run)

    with pytest.raises(RuntimeError) as exc_info:
        run_code("print('hello')")
    assert "failed to execute" in str(exc_info.value)
    assert "Cannot connect to the Docker daemon" in str(exc_info.value)


def test_container_timeout_cleanup(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "container")
    monkeypatch.setattr(config, "SANDBOX_ENGINE", "docker")
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/docker")

    called_cmds = []

    def mock_run(cmd, *args, **kwargs):
        called_cmds.append(cmd)
        if "rm" in cmd:
            mock_res = MagicMock()
            mock_res.returncode = 0
            return mock_res
        raise subprocess.TimeoutExpired(cmd, timeout=5)

    monkeypatch.setattr(subprocess, "run", mock_run)

    passed, output = run_code("print('hello')", timeout=5)
    assert passed is False
    assert "Timeout after 5s" in output

    assert len(called_cmds) == 2
    assert "run" in called_cmds[0]
    cleanup_cmd = called_cmds[1]
    assert cleanup_cmd[0] == "docker"
    assert cleanup_cmd[1] == "rm"
    assert cleanup_cmd[2] == "-f"
    run_container_name = called_cmds[0][called_cmds[0].index("--name") + 1]
    assert cleanup_cmd[3] == run_container_name


def test_container_oom_exit_code(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "container")
    monkeypatch.setattr(config, "SANDBOX_ENGINE", "docker")
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/docker")

    called_cmds = []

    def mock_run(cmd, *args, **kwargs):
        called_cmds.append(cmd)
        mock_res = MagicMock()
        mock_res.returncode = 137
        mock_res.stdout = ""
        mock_res.stderr = "Killed"
        return mock_res

    monkeypatch.setattr(subprocess, "run", mock_run)

    passed, output = run_code("print('hello')")
    assert passed is False
    assert "Memory limit exceeded" in output


def test_container_execution_failure_returns_contract(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "container")
    monkeypatch.setattr(config, "SANDBOX_ENGINE", "docker")
    monkeypatch.setattr(
        shutil,
        "which",
        lambda x: "/usr/bin/docker" if x == "docker" else None,
    )

    def mock_run(cmd, *args, **kwargs):
        mock_res = MagicMock()
        mock_res.returncode = 1
        mock_res.stdout = ""
        mock_res.stderr = "Traceback (most recent call last):\nValueError: boom"
        return mock_res

    monkeypatch.setattr(subprocess, "run", mock_run)

    passed, output = run_code("raise ValueError('boom')")
    assert passed is False
    assert "ValueError: boom" in output


def test_subprocess_backend_preserves_behavior(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "subprocess")

    called_cmds = []

    def mock_run(cmd, *args, **kwargs):
        called_cmds.append(cmd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "hello\n"
        mock_res.stderr = ""
        return mock_res

    monkeypatch.setattr(subprocess, "run", mock_run)

    passed, output = run_code("print('hello')")
    assert passed is True
    assert "hello" in output

    assert len(called_cmds) == 1
    # Verify it called host Python instead of container engine
    assert called_cmds[0][0] == sys.executable


def _is_engine_available() -> bool:
    engine = os.environ.get("WHETSTONE_TEST_CONTAINER_ENGINE", "docker")
    return shutil.which(engine) is not None


def _is_daemon_running() -> bool:
    engine = os.environ.get("WHETSTONE_TEST_CONTAINER_ENGINE", "docker")
    if not shutil.which(engine):
        return False
    try:
        res = subprocess.run([engine, "ps"], capture_output=True, timeout=3)
        return res.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(
    os.environ.get("WHETSTONE_INTEGRATION_TEST") != "1",
    reason="Integration tests disabled. Set WHETSTONE_INTEGRATION_TEST=1",
)
class TestContainerSandboxIntegration:
    @pytest.fixture(autouse=True)
    def setup_sandbox(self, monkeypatch):
        if not _is_engine_available() or not _is_daemon_running():
            pytest.skip(
                "Docker or Podman daemon is not available for integration tests."
            )

        engine = os.environ.get("WHETSTONE_TEST_CONTAINER_ENGINE", "docker")
        monkeypatch.setattr(config, "SANDBOX_BACKEND", "container")
        monkeypatch.setattr(config, "SANDBOX_ENGINE", engine)
        monkeypatch.setattr(config, "SANDBOX_IMAGE", "python:3.11-slim")
        monkeypatch.setattr(config, "SANDBOX_MEMORY_LIMIT", "256m")
        monkeypatch.setattr(config, "SANDBOX_CPU_LIMIT", 1.0)
        monkeypatch.setattr(config, "SANDBOX_NETWORK_ACCESS", False)

    def test_integration_good_code_passes(self):
        passed, output = run_code("print('hello from container')")
        assert passed is True
        assert "hello from container" in output

    def test_integration_bad_code_fails(self):
        passed, output = run_code("raise ValueError('boom')")
        assert passed is False
        assert "ValueError" in output

    def test_integration_syntax_error_fails(self):
        passed, output = run_code("def f(\n")
        assert passed is False

    def test_integration_timeout_kills_infinite_loop(self):
        passed, output = run_code("import time; time.sleep(10)", timeout=2)
        assert passed is False
        assert "Timeout" in output

    def test_integration_exit_code_nonzero_fails(self):
        passed, output = run_code("import sys; sys.exit(1)")
        assert passed is False

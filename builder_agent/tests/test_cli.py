import json
import os
import tempfile
from unittest.mock import patch

from builder_agent.cli import (
    EXIT_ABORTED,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    EXIT_USAGE,
    main,
)
from builder_agent.memory import Memory
from builder_agent.schemas import Plan, Spec, SubTask, Verdict


def _make_success_result():
    return {
        "succeeded": True,
        "halted_at": None,
        "plan": Plan(subtasks=[
            SubTask(id="t1", description="add", acceptance_criteria=["adds"]),
        ]),
        "spec": Spec(
            request="test",
            description="test",
            acceptance_criteria=["adds"],
            assumptions=[],
            output_type="python_module",
        ),
        "subtask_results": {},
        "artifact": "def add(a,b): return a+b",
        "final_verdict": Verdict(
            passed=True, score=9, tests_passed=True,
            issues=[], exec_output="ok",
        ),
        "aborted_reason": None,
        "usage": {
            "input_tokens": 100, "output_tokens": 50,
            "total_tokens": 150, "limit": 200000,
        },
    }


def _make_failure_result():
    r = _make_success_result()
    r["succeeded"] = False
    r["final_verdict"] = Verdict(
        passed=False, score=5, tests_passed=False,
        issues=["bad"], exec_output="err",
    )
    return r


def _make_aborted_result():
    r = _make_failure_result()
    r["aborted_reason"] = "token_budget"
    return r


# --- build command ---


@patch("builder_agent.cli.orchestrate", return_value=_make_success_result())
def test_build_success_exit_code(mock_orch):
    code = main(["build", "add function", "--non-interactive", "--no-memory"])
    assert code == EXIT_SUCCESS


@patch("builder_agent.cli.orchestrate", return_value=_make_failure_result())
def test_build_failure_exit_code(mock_orch):
    code = main(["build", "add function", "--non-interactive", "--no-memory"])
    assert code == EXIT_FAILURE


@patch("builder_agent.cli.orchestrate", return_value=_make_aborted_result())
def test_build_aborted_exit_code(mock_orch):
    code = main(["build", "add function", "--non-interactive", "--no-memory"])
    assert code == EXIT_ABORTED


@patch("builder_agent.cli.orchestrate", return_value=_make_success_result())
def test_build_json_output(mock_orch, capsys):
    main(["build", "test", "--non-interactive", "--no-memory", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["succeeded"] is True


@patch("builder_agent.cli.orchestrate", return_value=_make_success_result())
def test_build_non_interactive_flag(mock_orch):
    main(["build", "test", "--non-interactive", "--no-memory"])
    call_kwargs = mock_orch.call_args[1]
    assert call_kwargs["interactive"] is False


@patch("builder_agent.cli.orchestrate", return_value=_make_success_result())
def test_build_output_flag(mock_orch):
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        main([
            "build", "test", "--non-interactive",
            "--no-memory", "--output", path,
        ])
        with open(path) as f:
            content = f.read()
        assert "def add" in content
    finally:
        os.unlink(path)


@patch("builder_agent.cli.orchestrate", return_value=_make_success_result())
def test_build_no_memory_flag(mock_orch):
    main(["build", "test", "--non-interactive", "--no-memory"])
    call_kwargs = mock_orch.call_args[1]
    assert call_kwargs["memory"] is None


@patch("builder_agent.cli._repl", return_value=EXIT_SUCCESS)
def test_no_args_launches_repl(mock_repl):
    code = main([])
    assert code == EXIT_SUCCESS
    mock_repl.assert_called_once()


@patch("builder_agent.cli._repl", return_value=EXIT_SUCCESS)
def test_chat_command_launches_repl(mock_repl):
    code = main(["chat"])
    assert code == EXIT_SUCCESS
    mock_repl.assert_called_once()


# --- memory commands ---


class _StubEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 8


def test_memory_list(capsys):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from builder_agent.schemas import MemoryRecord
    mem = Memory(db_path=path, embedder=_StubEmbedder())
    mem.store(MemoryRecord(
        request="test", output_type="python_module",
        subtask_desc="do stuff", failures=[], fix_summary="ok",
        final_code="code", embedding=[0.1] * 8, record_type="subtask",
    ))
    with patch("builder_agent.cli.Memory", return_value=mem):
        code = main(["memory", "list"])
    assert code == EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "test" in out


def test_memory_show(capsys):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from builder_agent.schemas import MemoryRecord
    mem = Memory(db_path=path, embedder=_StubEmbedder())
    mem.store(MemoryRecord(
        request="test", output_type="python_module",
        subtask_desc="do stuff", failures=["err"],
        fix_summary="fixed it", final_code="code",
        embedding=[0.1] * 8, record_type="subtask",
    ))
    with patch("builder_agent.cli.Memory", return_value=mem):
        code = main(["memory", "show", "1"])
    assert code == EXIT_SUCCESS
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["fix_summary"] == "fixed it"


def test_memory_show_not_found(capsys):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    mem = Memory(db_path=path, embedder=_StubEmbedder())
    with patch("builder_agent.cli.Memory", return_value=mem):
        code = main(["memory", "show", "999"])
    assert code == EXIT_FAILURE


def test_memory_clear_yes(capsys):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    from builder_agent.schemas import MemoryRecord
    mem = Memory(db_path=path, embedder=_StubEmbedder())
    mem.store(MemoryRecord(
        request="test", output_type="python_module",
        subtask_desc="stuff", failures=[], fix_summary="ok",
        final_code="c", embedding=[0.1] * 8, record_type="subtask",
    ))
    with patch("builder_agent.cli.Memory", return_value=mem):
        code = main(["memory", "clear", "--yes"])
    assert code == EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "Deleted" in out
    assert len(mem.list_records()) == 0


def test_memory_subcommand_help():
    code = main(["memory"])
    assert code == EXIT_USAGE


# --- module entrypoint ---


def test_module_entrypoint():
    assert os.path.exists(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "__main__.py",
        )
    )


def _make_package_result():
    r = _make_success_result()
    r["spec"].output_type = "python_package"
    r["artifact"] = {
        "calculator/__init__.py": "from .core import add\n",
        "calculator/core.py": "def add(a, b): return a + b\n"
    }
    return r


@patch("builder_agent.cli.orchestrate", return_value=_make_package_result())
def test_build_output_dir_flag(mock_orch):
    with tempfile.TemporaryDirectory() as tmpdir:
        code = main([
            "build", "test", "--non-interactive",
            "--no-memory", "--output-dir", tmpdir,
        ])
        assert code == EXIT_SUCCESS
        assert os.path.exists(os.path.join(tmpdir, "calculator", "__init__.py"))
        assert os.path.exists(os.path.join(tmpdir, "calculator", "core.py"))

        with open(os.path.join(tmpdir, "calculator", "core.py")) as f:
            content = f.read()
        assert "def add" in content


@patch("builder_agent.cli.orchestrate")
def test_build_output_dir_path_traversal_rejection(mock_orch):
    r = _make_package_result()
    r["artifact"] = {"../unsafe.py": "bad"}
    mock_orch.return_value = r

    with tempfile.TemporaryDirectory() as tmpdir:
        import pytest
        with pytest.raises(ValueError) as exc_info:
            main([
                "build", "test", "--non-interactive",
                "--no-memory", "--output-dir", tmpdir,
            ])
        assert "Unsafe path traversal" in str(exc_info.value)


@patch("builder_agent.cli.orchestrate", return_value=_make_package_result())
def test_build_output_dir_omitted_default_fallback(mock_orch, monkeypatch):
    # Use temporary directory for default output folder path to prevent pollution
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override output_package default or patch writing folder
        dest_pkg = os.path.join(tmpdir, "default_out")

        # Intercept _print_result behavior or temporarily mock the folder resolution
        from builder_agent import cli as cli_mod
        orig_print_result = cli_mod._print_result

        def mock_print_result(result, output_path="", output_dir=""):
            # Redirect to dest_pkg
            return orig_print_result(
                result, output_path=output_path, output_dir=dest_pkg
            )

        monkeypatch.setattr(cli_mod, "_print_result", mock_print_result)

        code = main(["build", "test", "--non-interactive", "--no-memory"])
        assert code == EXIT_SUCCESS
        assert os.path.exists(os.path.join(dest_pkg, "calculator", "core.py"))
def test_repl_clarify_toggle(monkeypatch, capsys):
    inputs = [
        "/clarify",
        "/clarify off",
        "/clarify on",
        "/clarify toggle",
        "/quit",
    ]
    input_iter = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda *a: next(input_iter))

    from builder_agent.cli import _repl

    code = _repl()
    assert code == 0

    out = capsys.readouterr().out
    assert "Interactive clarification is currently on" in out
    assert "Interactive clarification is now off" in out
    assert "Interactive clarification is now on" in out
    assert "Interactive clarification is currently on" in out


@patch("builder_agent.cli.orchestrate")
@patch("builder_agent.cli.detect_ambiguity")
def test_repl_interactive_clarification_flow(
    mock_detect,
    mock_orch,
    monkeypatch,
):
    mock_detect.return_value = ["Q1?", "Q2?"]
    mock_orch.return_value = _make_success_result()

    inputs = [
        "build a calculator",
        "python",
        "",
        "/quit",
    ]
    input_iter = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda *a: next(input_iter))

    from builder_agent.cli import _repl

    code = _repl()
    assert code == 0

    mock_orch.assert_called_once()
    called_request = mock_orch.call_args[0][0]

    assert "build a calculator" in called_request
    assert "Clarifications:" in called_request
    assert "- Q: Q1?\n  A: python" in called_request
    assert "- Q: Q2?\n  A: Not specified" in called_request


@patch("builder_agent.cli.detect_ambiguity")
def test_repl_interactive_clarification_ctrl_c(
    mock_detect,
    monkeypatch,
    capsys,
):
    mock_detect.return_value = ["Q1?"]

    class MockIter:
        def __init__(self):
            self.step = 0

        def __call__(self, *args, **kwargs):
            self.step += 1
            if self.step == 1:
                return "build a calculator"
            if self.step == 2:
                raise KeyboardInterrupt()
            return "/quit"

    monkeypatch.setattr("builtins.input", MockIter())

    from builder_agent.cli import _repl

    code = _repl()
    assert code == 0

    out = capsys.readouterr().out
    assert "Build cancelled" in out


@patch("builder_agent.cli.orchestrate")
@patch("builder_agent.cli.detect_ambiguity")
def test_cli_build_interactive_clarify_flag(
    mock_detect,
    mock_orch,
    monkeypatch,
):
    mock_detect.return_value = ["Q1?"]
    mock_orch.return_value = _make_success_result()

    inputs = ["answers"]
    input_iter = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda *a: next(input_iter))

    code = main(
        [
            "build",
            "build a calculator",
            "--interactive-clarify",
            "--non-interactive",
        ]
    )

    assert code == 0
    mock_orch.assert_called_once()
    assert "answers" in mock_orch.call_args[0][0]


@patch("builder_agent.cli.detect_ambiguity")
def test_cli_build_interactive_clarify_flag_ctrl_c(
    mock_detect,
    monkeypatch,
):
    mock_detect.return_value = ["Q1?"]

    monkeypatch.setattr(
        "builtins.input",
        lambda *a: exec("raise KeyboardInterrupt()"),
    )

    code = main(
        [
            "build",
            "build a calculator",
            "--interactive-clarify",
            "--non-interactive",
        ]
    )

    assert code == 2


def test_cli_build_interactive_clarify_with_json(capsys):
    code = main(
        [
            "build",
            "req",
            "--interactive-clarify",
            "--json",
        ]
    )

    assert code == 3

    err = capsys.readouterr().err
    assert "cannot be used with --json" in err


def test_progress_renderer_handles_chunk(capsys):
    from builder_agent.cli import ProgressRenderer, Spinner

    spinner = Spinner()
    renderer = ProgressRenderer(spinner)

    renderer("generating", {"iteration": 1, "subtask": "t1"})
    renderer("chunk", {"chunk": "def add(a, b):\n"})
    renderer("chunk", {"chunk": "    return a + b"})
    renderer("critiquing", {})

    captured = capsys.readouterr().out

    assert "Generating iter 1:" in captured
    assert "    def add(a, b):\n        return a + b" in captured


@patch("builder_agent.cli.input")
@patch("builder_agent.cli.orchestrate")
@patch("builder_agent.cli.detect_ambiguity")
def test_repl_export_and_history(
    mock_detect, mock_orchestrate, mock_input, tmp_path, capsys
):
    mock_detect.return_value = []
    mock_orchestrate.return_value = _make_success_result()
    mock_input.side_effect = [
        "/export",
        "/history",
        "build request",
        "/history",
        "/history 1",
        "/history 2",
        "/history abc",
        "/export",
        "/export subdir/custom.py",
        "/quit"
    ]

    import os
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        from builder_agent.cli import _repl
        _repl()
    finally:
        os.chdir(orig_cwd)

    captured = capsys.readouterr().out

    # 1. /export (no build yet)
    assert "No build artifact available to export yet." in captured

    # 2. /history (no builds yet)
    assert "No builds have been performed in this session yet." in captured

    # 3. build summary /history
    assert "Build History" in captured
    assert "#1   build request" in captured

    # 4. build detailed /history 1
    assert "Build #1" in captured
    assert "Request    build request" in captured
    assert "t1: add" in captured
    assert "def add(a,b): return a+b" in captured

    # 5. build detailed /history 2 (out of range)
    assert "Invalid build number. Range: 1 to 1." in captured

    # 6. build detailed /history abc (invalid number)
    assert "Invalid build number. Please specify an integer." in captured

    # 7. /export (default artifact.py)
    assert os.path.exists(tmp_path / "artifact.py")
    with open(tmp_path / "artifact.py") as f:
        assert f.read() == "def add(a,b): return a+b"

    # 8. /export custom (subdir/custom.py)
    assert os.path.exists(tmp_path / "subdir" / "custom.py")
    with open(tmp_path / "subdir" / "custom.py") as f:
        assert f.read() == "def add(a,b): return a+b"

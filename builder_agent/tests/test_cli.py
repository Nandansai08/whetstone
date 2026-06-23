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

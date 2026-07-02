from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from builder_agent.web.app import app
from builder_agent.web.history import BuildHistory


@pytest.fixture
def temp_history_db(tmp_path):
    db_file = tmp_path / "test_history.db"
    return BuildHistory(db_path=str(db_file))

def test_build_history_db_methods(temp_history_db):
    history = temp_history_db
    build_id = history.create_build("build request", "python_module")
    assert build_id is not None

    history.add_attempt(
        build_id=build_id,
        subtask_id="t1",
        iteration=1,
        code="def add(a, b): return a + b",
        score=9,
        passed=True,
        issues=["minor warning"]
    )

    builds = history.get_builds()
    assert len(builds) == 1
    assert builds[0]["request"] == "build request"
    assert builds[0]["status"] == "running"

    attempts = history.get_attempts(build_id)
    assert len(attempts) == 1
    assert attempts[0]["subtask_id"] == "t1"
    assert attempts[0]["score"] == 9
    assert attempts[0]["issues"] == ["minor warning"]

    history.update_build_status(build_id, "passed", score=9, artifact="code")
    build = history.get_build(build_id)
    assert build["status"] == "passed"
    assert build["score"] == 9
    assert build["artifact"] == "code"

def test_web_routes(temp_history_db):
    with patch("builder_agent.web.routes.BuildHistory", return_value=temp_history_db):
        client = TestClient(app)

        # Test index route
        response = client.get("/")
        assert response.status_code == 200
        assert "Build History" in response.text

        # Create a build
        build_id = temp_history_db.create_build("test request", "python_module")
        temp_history_db.add_attempt(build_id, "t1", 1, "code", 9, True, [])
        temp_history_db.update_build_status(build_id, "passed", 9, "final_code")

        # Test detail route
        response = client.get(f"/build/{build_id}")
        assert response.status_code == 200
        assert "test request" in response.text
        assert "t1" in response.text

        # Test diff route
        temp_history_db.add_attempt(build_id, "t1", 2, "new_code", 10, True, [])
        response = client.get(f"/build/{build_id}/diff?subtask=t1&iter1=1&iter2=2")
        assert response.status_code == 200
        assert "Iteration 1" in response.text

@pytest.mark.anyio
async def test_stream_sse_endpoint(temp_history_db):
    with patch("builder_agent.web.routes.BuildHistory", return_value=temp_history_db):
        build_id = temp_history_db.create_build("test sse", "python_module")
        temp_history_db.add_attempt(build_id, "t1", 1, "code", 9, True, [])
        temp_history_db.update_build_status(build_id, "passed", 9, "final_code")

        from builder_agent.web.routes import get_stream
        response = await get_stream(build_id)

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert any("event: attempt" in c for c in chunks)
        assert any("event: status" in c for c in chunks)

def test_cli_web_subcommand():
    from builder_agent.cli import main
    with patch("builder_agent.web.app.start_server") as mock_start:
        code = main(["web", "--port", "9000"])
        assert code == 0
        mock_start.assert_called_once_with(host="127.0.0.1", port=9000)


@patch("builder_agent.orchestrate.checkpoint.build_id", return_value="test_bid")
@patch("builder_agent.orchestrate.checkpoint.load", return_value=None)
@patch("builder_agent.orchestrate.checkpoint.save")
@patch("builder_agent.orchestrate.clarify")
@patch("builder_agent.orchestrate.make_plan")
@patch("builder_agent.orchestrate._async_orchestrate", new_callable=MagicMock)
@patch("builder_agent.orchestrate._run_async")
@patch("builder_agent.orchestrate.integrate")
@patch("builder_agent.orchestrate.verify")
def test_orchestrate_web_history_opt_in(
    mock_verify, mock_integrate, mock_run_async, mock_async_orch,
    mock_make_plan, mock_clarify, mock_save, mock_load, mock_build_id
):
    from builder_agent.orchestrate import orchestrate
    from builder_agent.schemas import Plan, Spec, Verdict

    mock_clarify.return_value = Spec(
        request="test prompt",
        description="desc",
        acceptance_criteria=[],
        assumptions=[],
        output_type="python_module"
    )
    mock_make_plan.return_value = Plan(subtasks=[])
    mock_run_async.return_value = {
        "succeeded": True,
        "outputs": {},
        "subtask_results": {}
    }
    mock_integrate.return_value = "code"
    mock_verify.return_value = Verdict(
        passed=True,
        score=10,
        tests_passed=True,
        issues=[],
        exec_output=""
    )

    # 1. By default, enable_web_history is False, BuildHistory not called
    with patch("builder_agent.web.history.BuildHistory") as mock_history_class:
        orchestrate("test prompt")
        mock_history_class.assert_not_called()

    # 2. When enable_web_history is True, BuildHistory is instantiated
    with patch("builder_agent.web.history.BuildHistory") as mock_history_class:
        mock_hist_inst = mock_history_class.return_value
        mock_hist_inst.create_build.return_value = "web_build_123"

        orchestrate("test prompt", enable_web_history=True)
        mock_history_class.assert_called_once()
        mock_hist_inst.create_build.assert_called_once()

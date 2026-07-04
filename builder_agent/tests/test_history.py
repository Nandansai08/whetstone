from builder_agent import config
from builder_agent.web.history import BuildHistory


def test_prune_empty_db(tmp_path):
    history = BuildHistory(str(tmp_path / "history.db"))
    history.prune()


def test_prune_age_removes_old_build(tmp_path):
    history = BuildHistory(str(tmp_path / "history.db"))
    with history._connect() as conn:
        conn.execute(
            "INSERT INTO builds (id, request, output_type, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("old-build", "test", "text", "done", "2000-01-01T00:00:00+00:00"),
        )
    history.prune()

    builds = history.get_builds()
    assert builds == []


def test_prune_count_removes_old_builds(tmp_path):
    history = BuildHistory(str(tmp_path / "history.db"))

    old_max = config.MAX_BUILDS
    config.MAX_BUILDS = 2

    try:
        for i in range(5):
            history.create_build(f"request-{i}", "text")
            history.update_build_status(
                build_id=history.get_builds()[0]["id"],
                status="done",
            )

        history.prune()

        builds = history.get_builds()

        assert len(builds) == 2
        assert builds[0]["request"] == "request-4"
        assert builds[1]["request"] == "request-3"

    finally:
        config.MAX_BUILDS = old_max

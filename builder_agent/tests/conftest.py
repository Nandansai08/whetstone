import pytest

from builder_agent import config


@pytest.fixture(autouse=True)
def _isolate_checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))


@pytest.fixture(autouse=True)
def disable_plugins_for_non_plugin_tests(request, monkeypatch):
    if "test_plugins" not in request.module.__name__:
        monkeypatch.setattr(config, "PLUGINS_DISABLED", ["LinterPlugin"])
        monkeypatch.setattr(config, "PLUGIN_DIR", "non_existent_directory_for_testing")

import pytest

from builder_agent import config


@pytest.fixture(autouse=True)
def _isolate_checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))

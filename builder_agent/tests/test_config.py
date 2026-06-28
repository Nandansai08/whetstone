import importlib
from unittest.mock import patch

import pytest

from builder_agent import config
from builder_agent.cli import EXIT_SUCCESS, main
from builder_agent.config import ModelConfig


@pytest.fixture(autouse=True)
def restore_config_defaults():
    # Store standard defaults
    defaults = {
        "MAX_ITERATIONS": 4,
        "SCORE_THRESHOLD": 8,
        "PLATEAU_PATIENCE": 2,
        "EXEC_TIMEOUT": 10,
        "TOKEN_BUDGET": 200_000,
        "MEMORY_DB_PATH": "./builder_memory.db",
        "MEMORY_TOP_K": 3,
        "MEMORY_MIN_SIMILARITY": 0.4,
        "EMBEDDER": "tfidf",
        "MAX_SUBTASKS": 5,
        "MAX_RETRIES": 3,
        "RETRY_DELAY": 1.0,
        "SANDBOX_BACKEND": "subprocess",
        "SANDBOX_ENGINE": "docker",
        "SANDBOX_IMAGE": "python:3.11-slim",
        "SANDBOX_MEMORY_LIMIT": "256m",
        "SANDBOX_CPU_LIMIT": 1.0,
        "SANDBOX_NETWORK_ACCESS": False,
        "WORKER_MODEL": ModelConfig(
            "openai",
            "meta-llama/llama-4-scout",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1"
        ),
        "JUDGE_MODEL": ModelConfig(
            "openai",
            "google/gemini-2.5-flash-preview",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1"
        ),
        "PLANNER_MODEL": ModelConfig(
            "openai",
            "meta-llama/llama-4-scout",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1"
        ),
        "ESCALATION_MODEL": ModelConfig(
            "openai",
            "google/gemini-2.5-flash-preview",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1"
        ),
    }

    yield

    # Restore defaults
    for k, v in defaults.items():
        setattr(config, k, v)


def test_default_behavior_no_config(tmp_path):
    with patch("pathlib.Path.home", return_value=tmp_path / "home"), \
         patch("pathlib.Path.cwd", return_value=tmp_path / "project"):

         importlib.reload(config)

         assert config.MAX_ITERATIONS == 4
         assert config.WORKER_MODEL.provider == "openai"


def test_global_config_only(tmp_path):
    home_dir = tmp_path / "home"
    config_dir = home_dir / ".config" / "whetstone"
    config_dir.mkdir(parents=True)

    global_toml = """
max_iterations = 42
score_threshold = 9
"""
    with open(config_dir / "config.toml", "w") as f:
        f.write(global_toml)

    with patch("pathlib.Path.home", return_value=home_dir), \
         patch("pathlib.Path.cwd", return_value=tmp_path / "project"):

         importlib.reload(config)

         assert config.MAX_ITERATIONS == 42
         assert config.SCORE_THRESHOLD == 9


def test_local_config_overrides_global(tmp_path):
    home_dir = tmp_path / "home"
    config_dir = home_dir / ".config" / "whetstone"
    config_dir.mkdir(parents=True)

    with open(config_dir / "config.toml", "w") as f:
        f.write("max_iterations = 42\nscore_threshold = 9")

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with open(project_dir / ".whetstone.toml", "w") as f:
        f.write("max_iterations = 10\nscore_threshold = 7\nplateau_patience = 5")

    with patch("pathlib.Path.home", return_value=home_dir), \
         patch("pathlib.Path.cwd", return_value=project_dir):

         importlib.reload(config)

         assert config.MAX_ITERATIONS == 10
         assert config.SCORE_THRESHOLD == 7
         assert config.PLATEAU_PATIENCE == 5


def test_model_configuration_overrides(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    toml = """
[models.worker]
provider = "anthropic"
model_id = "claude-3-opus"
api_key_env = "MY_KEY"
base_url = "https://custom.endpoint"
"""
    with open(project_dir / ".whetstone.toml", "w") as f:
        f.write(toml)

    with patch("pathlib.Path.home", return_value=tmp_path / "home"), \
         patch("pathlib.Path.cwd", return_value=project_dir):

         importlib.reload(config)

         assert config.WORKER_MODEL.provider == "anthropic"
         assert config.WORKER_MODEL.model_id == "claude-3-opus"
         assert config.WORKER_MODEL.api_key_env == "MY_KEY"
         assert config.WORKER_MODEL.base_url == "https://custom.endpoint"


def test_invalid_toml_raises(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with open(project_dir / ".whetstone.toml", "w") as f:
        f.write("invalid = toml = format")

    with patch("pathlib.Path.home", return_value=tmp_path / "home"), \
         patch("pathlib.Path.cwd", return_value=project_dir):

         with pytest.raises(RuntimeError) as exc_info:
             importlib.reload(config)
         assert "Failed to parse configuration file" in str(exc_info.value)
         assert ".whetstone.toml" in str(exc_info.value)


def test_cli_init(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    with patch("pathlib.Path.cwd", return_value=project_dir):
        rc = main(["init"])
        assert rc == EXIT_SUCCESS
        config_file = project_dir / ".whetstone.toml"
        assert config_file.exists()

        content = config_file.read_text(encoding="utf-8")
        assert "max_iterations = 4" in content
        assert "[models.worker]" in content

        rc = main(["init"])
        assert rc == EXIT_SUCCESS


def test_model_config_fields():
    m = ModelConfig("openai", "gpt-4o", "MY_KEY", "http://localhost:8000")
    assert m.provider == "openai"
    assert m.model_id == "gpt-4o"
    assert m.api_key_env == "MY_KEY"
    assert m.base_url == "http://localhost:8000"


def test_model_config_defaults():
    m = ModelConfig("anthropic", "claude-sonnet-4-6")
    assert m.api_key_env == ""
    assert m.base_url == ""


def test_worker_model_is_model_config():
    assert isinstance(config.WORKER_MODEL, ModelConfig)
    assert config.WORKER_MODEL.provider in ("anthropic", "openai")


def test_judge_differs_from_worker():
    assert config.JUDGE_MODEL.model_id != config.WORKER_MODEL.model_id


def test_thresholds_are_positive():
    assert config.MAX_ITERATIONS > 0
    assert config.SCORE_THRESHOLD > 0
    assert config.PLATEAU_PATIENCE > 0
    assert config.EXEC_TIMEOUT > 0
    assert config.TOKEN_BUDGET > 0
    assert config.MEMORY_TOP_K > 0


def test_score_threshold_in_range():
    assert 1 <= config.SCORE_THRESHOLD <= 10

from builder_agent.config import JUDGE_MODEL, WORKER_MODEL, ModelConfig


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
    assert isinstance(WORKER_MODEL, ModelConfig)
    assert WORKER_MODEL.provider in ("anthropic", "openai")


def test_judge_differs_from_worker():
    assert JUDGE_MODEL.model_id != WORKER_MODEL.model_id


def test_thresholds_are_positive():
    from builder_agent import config
    assert config.MAX_ITERATIONS > 0
    assert config.SCORE_THRESHOLD > 0
    assert config.PLATEAU_PATIENCE > 0
    assert config.EXEC_TIMEOUT > 0
    assert config.TOKEN_BUDGET > 0
    assert config.MEMORY_TOP_K > 0


def test_score_threshold_in_range():
    from builder_agent import config
    assert 1 <= config.SCORE_THRESHOLD <= 10

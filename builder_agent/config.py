from dataclasses import dataclass


@dataclass
class ModelConfig:
    provider: str  # "anthropic" | "openai" | any registered provider
    model_id: str
    api_key_env: str = ""  # env var name; empty = provider default
    base_url: str = ""  # custom endpoint (Ollama, vLLM, Azure, etc.)


_OPENROUTER = "https://openrouter.ai/api/v1"
_OR_KEY = "OPENROUTER_API_KEY"

WORKER_MODEL = ModelConfig(
    "openai", "meta-llama/llama-4-scout",
    api_key_env=_OR_KEY, base_url=_OPENROUTER,
)
JUDGE_MODEL = ModelConfig(
    "openai", "google/gemini-2.5-flash-preview",
    api_key_env=_OR_KEY, base_url=_OPENROUTER,
)
PLANNER_MODEL = ModelConfig(
    "openai", "meta-llama/llama-4-scout",
    api_key_env=_OR_KEY, base_url=_OPENROUTER,
)
ESCALATION_MODEL = ModelConfig(
    "openai", "google/gemini-2.5-flash-preview",
    api_key_env=_OR_KEY, base_url=_OPENROUTER,
)
MAX_ITERATIONS = 4
SCORE_THRESHOLD = 8
PLATEAU_PATIENCE = 2
EXEC_TIMEOUT = 10
TOKEN_BUDGET = 200_000
MEMORY_DB_PATH = "./builder_memory.db"
MEMORY_TOP_K = 3
MEMORY_MIN_SIMILARITY = 0.4
EMBEDDER = "tfidf"
MAX_SUBTASKS = 5
INTERACTIVE_CLARIFY = True

# Sandbox Configuration
SANDBOX_BACKEND = "subprocess"      # "subprocess" | "container"
SANDBOX_ENGINE = "docker"           # "docker" | "podman"
SANDBOX_IMAGE = "python:3.11-slim"
SANDBOX_MEMORY_LIMIT = "256m"
SANDBOX_CPU_LIMIT = 1.0
SANDBOX_NETWORK_ACCESS = False


def _load_and_apply_config() -> None:
    import pathlib
    import tomllib

    global WORKER_MODEL, JUDGE_MODEL, PLANNER_MODEL, ESCALATION_MODEL
    global MAX_ITERATIONS, SCORE_THRESHOLD, PLATEAU_PATIENCE, EXEC_TIMEOUT
    global TOKEN_BUDGET, MEMORY_DB_PATH, MEMORY_TOP_K, MEMORY_MIN_SIMILARITY
    global EMBEDDER, MAX_SUBTASKS, MAX_RETRIES, RETRY_DELAY
    global SANDBOX_BACKEND, SANDBOX_ENGINE, SANDBOX_IMAGE
    global SANDBOX_MEMORY_LIMIT, SANDBOX_CPU_LIMIT, SANDBOX_NETWORK_ACCESS

    user_config_path = pathlib.Path.home() / ".config" / "whetstone" / "config.toml"
    project_config_path = pathlib.Path.cwd() / ".whetstone.toml"

    paths_to_load = []
    if user_config_path.exists():
        paths_to_load.append(user_config_path)
    if project_config_path.exists():
        paths_to_load.append(project_config_path)

    for path in paths_to_load:
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse configuration file '{path}': {e}"
            ) from e

        key_map = {
            "max_iterations": "MAX_ITERATIONS",
            "score_threshold": "SCORE_THRESHOLD",
            "plateau_patience": "PLATEAU_PATIENCE",
            "exec_timeout": "EXEC_TIMEOUT",
            "token_budget": "TOKEN_BUDGET",
            "memory_db_path": "MEMORY_DB_PATH",
            "memory_top_k": "MEMORY_TOP_K",
            "memory_min_similarity": "MEMORY_MIN_SIMILARITY",
            "embedder": "EMBEDDER",
            "max_subtasks": "MAX_SUBTASKS",
            "max_retries": "MAX_RETRIES",
            "retry_delay": "RETRY_DELAY",
            "sandbox_backend": "SANDBOX_BACKEND",
            "sandbox_engine": "SANDBOX_ENGINE",
            "sandbox_image": "SANDBOX_IMAGE",
            "sandbox_memory_limit": "SANDBOX_MEMORY_LIMIT",
            "sandbox_cpu_limit": "SANDBOX_CPU_LIMIT",
            "sandbox_network_access": "SANDBOX_NETWORK_ACCESS",
        }

        for toml_key, global_name in key_map.items():
            if toml_key in data:
                globals()[global_name] = data[toml_key]

        if "sandbox" in data and isinstance(data["sandbox"], dict):
            sb = data["sandbox"]
            sandbox_keys = {
                "backend": "SANDBOX_BACKEND",
                "engine": "SANDBOX_ENGINE",
                "image": "SANDBOX_IMAGE",
                "memory_limit": "SANDBOX_MEMORY_LIMIT",
                "cpu_limit": "SANDBOX_CPU_LIMIT",
                "network_access": "SANDBOX_NETWORK_ACCESS",
            }
            for toml_key, global_name in sandbox_keys.items():
                if toml_key in sb:
                    globals()[global_name] = sb[toml_key]

        if "memory" in data and isinstance(data["memory"], dict):
            m_data = data["memory"]
            memory_keys = {
                "db_path": "MEMORY_DB_PATH",
                "top_k": "MEMORY_TOP_K",
                "min_similarity": "MEMORY_MIN_SIMILARITY",
            }
            for toml_key, global_name in memory_keys.items():
                if toml_key in m_data:
                    globals()[global_name] = m_data[toml_key]

        if "models" in data and isinstance(data["models"], dict):
            models = data["models"]
            model_keys = {
                "worker": "WORKER_MODEL",
                "judge": "JUDGE_MODEL",
                "planner": "PLANNER_MODEL",
                "escalation": "ESCALATION_MODEL",
            }
            for toml_key, global_name in model_keys.items():
                if toml_key in models and isinstance(models[toml_key], dict):
                    m_conf = models[toml_key]
                    existing = globals()[global_name]
                    globals()[global_name] = ModelConfig(
                        provider=m_conf.get("provider", existing.provider),
                        model_id=m_conf.get("model_id", existing.model_id),
                        api_key_env=m_conf.get("api_key_env", existing.api_key_env),
                        base_url=m_conf.get("base_url", existing.base_url),
                    )


_load_and_apply_config()


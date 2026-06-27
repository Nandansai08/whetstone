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

# Sandbox Configuration
SANDBOX_BACKEND = "subprocess"      # "subprocess" | "container"
SANDBOX_ENGINE = "docker"           # "docker" | "podman"
SANDBOX_IMAGE = "python:3.11-slim"
SANDBOX_MEMORY_LIMIT = "256m"
SANDBOX_CPU_LIMIT = 1.0
SANDBOX_NETWORK_ACCESS = False


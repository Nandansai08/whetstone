"""Provider-agnostic LLM query, streaming, and embedding wrapper."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Callable, Generator

from builder_agent import config
from builder_agent.budget import TokenBudget
from builder_agent.config import ModelConfig

for _name in ("httpx", "httpcore", "openai", "anthropic"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.WARNING)
    _lg.propagate = False

logger = logging.getLogger(__name__)

_providers: dict[str, Callable] = {}
_stream_providers: dict[str, Callable] = {}
_embed_providers: dict[str, Callable] = {}
_budget = None
_progress_callback = None


def set_progress_callback(callback) -> None:
    global _progress_callback
    _progress_callback = callback


def get_progress_callback():
    return _progress_callback


def set_budget(budget: TokenBudget | None) -> None:
    """Set the active TokenBudget object for token cost tracking.

    Args:
        budget: TokenBudget instance or None to disable tracking.
    """
    global _budget
    _budget = budget


def get_budget() -> TokenBudget | None:
    """Retrieve the active TokenBudget instance.

    Returns:
        The active TokenBudget instance, or None if budget tracking is disabled.
    """
    return _budget


def _record_usage(
    input_tokens: int,
    output_tokens: int,
    model: ModelConfig | None = None,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> None:
    if _budget is not None:
        cost = None
        if model is not None:
            from builder_agent.config import MODEL_PRICING
            pricing = MODEL_PRICING.get(model.model_id)
            if pricing is not None:
                cost = (
                    (input_tokens / 1_000_000.0) * pricing["input"]
                    + (output_tokens / 1_000_000.0) * pricing["output"]
                )
        _budget.record(
            input_tokens,
            output_tokens,
            cost=cost,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )


def strip_fences(text: str) -> str:
    """Strip markdown code block fences (e.g. ```python) from LLM output.

    Args:
        text: Raw response string containing code block fences.

    Returns:
        Clean source code string without markdown surrounds.
    """
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def extract_json(text: str) -> str:
    """Scan response text and extract the first valid nested JSON block or array.

    Args:
        text: Raw text string potentially containing mixed prose and JSON.

    Returns:
        The extracted clean JSON string.
    """
    text = strip_fences(text)
    obj_start = text.find('{')
    arr_start = text.find('[')
    candidates = []
    if obj_start != -1:
        candidates.append((obj_start, '{', '}'))
    if arr_start != -1:
        candidates.append((arr_start, '[', ']'))
    candidates.sort(key=lambda x: x[0])

    for start, start_char, end_char in candidates:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    json.loads(candidate)
                    return candidate
    return text


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False
    if isinstance(exc, ConnectionError):
        return True

    # OpenAI exceptions
    try:
        import openai
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            if exc.status_code in (429, 500, 501, 502, 503, 504):
                return True
            return False
    except ImportError:
        pass

    # Anthropic exceptions
    try:
        import anthropic
        if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
            return True
        if isinstance(exc, anthropic.APIStatusError):
            if exc.status_code in (429, 500, 501, 502, 503, 504):
                return True
            return False
    except ImportError:
        pass

    # Generic httpx exceptions
    try:
        import httpx
        if isinstance(
            exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout)
        ):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code in (429, 500, 501, 502, 503, 504):
                return True
    except ImportError:
        pass

    return False


def _execute_with_retry(fn: Callable, *args, **kwargs):
    max_retries = getattr(config, "MAX_RETRIES", 3)
    base_delay = getattr(config, "RETRY_DELAY", 1.0)

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt < max_retries and _is_transient_error(exc):
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "LLM call failed with transient error: %s. "
                    "Retrying in %.1fs (attempt %d/%d)...",
                    exc, delay, attempt + 1, max_retries
                )
                cb = get_progress_callback()
                if cb is not None:
                    cb("retry", {
                        "attempt": attempt + 1,
                        "delay": delay,
                        "error": str(exc),
                    })
                time.sleep(delay)
            else:
                raise


def _execute_stream_with_retry(
    fn: Callable, *args, **kwargs
) -> Generator[str, None, None]:
    max_retries = getattr(config, "MAX_RETRIES", 3)
    base_delay = getattr(config, "RETRY_DELAY", 1.0)

    for attempt in range(max_retries + 1):
        try:
            gen = fn(*args, **kwargs)
            iterator = iter(gen)
            first_chunk = next(iterator)
            break
        except StopIteration:
            return
        except Exception as exc:
            if attempt < max_retries and _is_transient_error(exc):
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "LLM stream call failed with transient error: %s. "
                    "Retrying in %.1fs (attempt %d/%d)...",
                    exc, delay, attempt + 1, max_retries
                )
                cb = get_progress_callback()
                if cb is not None:
                    cb("retry", {
                        "attempt": attempt + 1,
                        "delay": delay,
                        "error": str(exc),
                    })
                time.sleep(delay)
            else:
                raise
    else:
        return

    yield first_chunk
    yield from iterator


def register_provider(name: str, fn: Callable) -> None:
    """Register a custom LLM text completion provider function.

    Args:
        name: Name identifier for the provider.
        fn: Callback function invoked for provider text completion calls.
    """
    _providers[name] = fn



def register_stream_provider(name: str, fn: Callable) -> None:
    """Register a custom LLM streaming completion provider function.

    Args:
        name: Name identifier for the provider.
        fn: Callback function yielding chunks of text.
    """
    _stream_providers[name] = fn


def register_embed_provider(name: str, fn: Callable) -> None:
    """Register a custom LLM text embedding provider function.

    Args:
        name: Name identifier for the provider.
        fn: Callback function returning lists of floats.
    """
    _embed_providers[name] = fn


def embed(
    text: str,
    *,
    model: ModelConfig,
) -> list[float]:
    """Generate text embedding vector using the configured model provider.

    Args:
        text: String of text to embed.
        model: Target model configuration profile.

    Returns:
        A list of floats representing the embedding vector.
    """
    fn = _embed_providers.get(model.provider)
    if fn is None:
        fn = _default_embed_provider(model.provider)
    return _execute_with_retry(fn, text, model=model)


def ask(
    prompt: str,
    *,
    model: ModelConfig,
    system: str = "",
    max_tokens: int = 4096,
) -> str:
    """Send text completion request to configuration model provider.

    Args:
        prompt: Raw instructions query to model.
        model: Configuration of target LLM model.
        system: System instructions instructions context block.
        max_tokens: Maximum response tokens permitted.

    Returns:
        The response content string.
    """
    fn = _providers.get(model.provider)
    if fn is None:
        fn = _default_provider(model.provider)
    return _execute_with_retry(
        fn, prompt, model=model, system=system, max_tokens=max_tokens
    )


def ask_stream(
    prompt: str,
    *,
    model: ModelConfig,
    system: str = "",
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    """Send streaming text completion request to model provider.

    Args:
        prompt: Instructions query.
        model: Configuration of target LLM.
        system: System instructions instructions block.
        max_tokens: Maximum tokens permitted.

    Yields:
        Chunks of completion response tokens.
    """
    fn = _stream_providers.get(model.provider)
    if fn is None:
        if model.provider in ("anthropic", "openai"):
            fn = _default_stream_provider(model.provider)
        else:
            # Fallback to ask() and yield the entire response as a single chunk
            yield ask(prompt, model=model, system=system, max_tokens=max_tokens)
            return
    yield from _execute_stream_with_retry(
        fn, prompt, model=model, system=system, max_tokens=max_tokens
    )


def _default_provider(name: str) -> Callable:
    if name == "anthropic":
        register_provider("anthropic", _ask_anthropic)
        return _ask_anthropic
    if name == "openai":
        register_provider("openai", _ask_openai)
        return _ask_openai
    raise ValueError(
        f"Unknown provider '{name}'. "
        f"Use register_provider() or pick 'anthropic'/'openai'."
    )


def _ask_anthropic(
    prompt: str,
    *,
    model: ModelConfig,
    system: str = "",
    max_tokens: int = 4096,
) -> str:
    import anthropic

    kwargs: dict = {}
    env_var = model.api_key_env or "ANTHROPIC_API_KEY"
    api_key = os.environ.get(env_var)
    if api_key:
        kwargs["api_key"] = api_key
    elif not model.base_url:
        raise RuntimeError(
            f"No API key found. Set {env_var} in your environment "
            f"or create a .env file. See .env.example."
        )
    if model.base_url:
        kwargs["base_url"] = model.base_url

    client = anthropic.Anthropic(**kwargs)
    msg_kwargs: dict = {
        "model": model.model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        msg_kwargs["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"}
            }
        ]
    response = client.messages.create(**msg_kwargs)
    if hasattr(response, "usage") and response.usage:
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        total_input = response.usage.input_tokens + cache_read + cache_create
        _record_usage(
            total_input,
            response.usage.output_tokens,
            model=model,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )
    return response.content[0].text or ""


def _ask_openai(
    prompt: str,
    *,
    model: ModelConfig,
    system: str = "",
    max_tokens: int = 4096,
) -> str:
    import openai

    kwargs: dict = {}
    env_var = model.api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(env_var)
    if api_key:
        kwargs["api_key"] = api_key
    elif model.base_url and "localhost" in model.base_url:
        kwargs["api_key"] = "ollama"
    else:
        raise RuntimeError(
            f"No API key found. Set {env_var} in your environment "
            f"or create a .env file. See .env.example."
        )
    if model.base_url:
        kwargs["base_url"] = model.base_url

    client = openai.OpenAI(**kwargs)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model.model_id,
        messages=messages,
        max_tokens=max_tokens,
    )
    if hasattr(response, "usage") and response.usage:
        cache_read = 0
        details = getattr(response.usage, "prompt_tokens_details", None)
        if details is not None:
            if isinstance(details, dict):
                cache_read = details.get("cached_tokens", 0) or 0
            else:
                cache_read = getattr(details, "cached_tokens", 0) or 0
        _record_usage(
            response.usage.prompt_tokens or 0,
            response.usage.completion_tokens or 0,
            model=model,
            cache_read_tokens=cache_read,
        )
    return response.choices[0].message.content or ""


def _default_stream_provider(name: str) -> Callable:
    if name == "anthropic":
        register_stream_provider("anthropic", _ask_stream_anthropic)
        return _ask_stream_anthropic
    if name == "openai":
        register_stream_provider("openai", _ask_stream_openai)
        return _ask_stream_openai
    raise ValueError(
        f"Unknown stream provider '{name}'."
    )


def _ask_stream_anthropic(
    prompt: str,
    *,
    model: ModelConfig,
    system: str = "",
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    import anthropic

    kwargs: dict = {}
    env_var = model.api_key_env or "ANTHROPIC_API_KEY"
    api_key = os.environ.get(env_var)
    if api_key:
        kwargs["api_key"] = api_key
    elif not model.base_url:
        raise RuntimeError(
            f"No API key found. Set {env_var} in your environment "
            f"or create a .env file. See .env.example."
        )
    if model.base_url:
        kwargs["base_url"] = model.base_url

    client = anthropic.Anthropic(**kwargs)
    msg_kwargs: dict = {
        "model": model.model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        msg_kwargs["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"}
            }
        ]

    with client.messages.stream(**msg_kwargs) as stream:
        for text in stream.text_stream:
            yield text
        message = stream.get_final_message()
        if message and hasattr(message, "usage") and message.usage:
            cache_read = getattr(message.usage, "cache_read_input_tokens", 0) or 0
            cache_create = getattr(message.usage, "cache_creation_input_tokens", 0) or 0
            total_input = message.usage.input_tokens + cache_read + cache_create
            _record_usage(
                total_input,
                message.usage.output_tokens,
                model=model,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_create,
            )


def _ask_stream_openai(
    prompt: str,
    *,
    model: ModelConfig,
    system: str = "",
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    import openai

    kwargs: dict = {}
    env_var = model.api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(env_var)
    if api_key:
        kwargs["api_key"] = api_key
    elif model.base_url and "localhost" in model.base_url:
        kwargs["api_key"] = "ollama"
    else:
        raise RuntimeError(
            f"No API key found. Set {env_var} in your environment "
            f"or create a .env file. See .env.example."
        )
    if model.base_url:
        kwargs["base_url"] = model.base_url

    client = openai.OpenAI(**kwargs)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model.model_id,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
    except TypeError:
        response = client.chat.completions.create(
            model=model.model_id,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )

    for chunk in response:
        if hasattr(chunk, "usage") and chunk.usage:
            cache_read = 0
            details = getattr(chunk.usage, "prompt_tokens_details", None)
            if details is not None:
                if isinstance(details, dict):
                    cache_read = details.get("cached_tokens", 0) or 0
                else:
                    cache_read = getattr(details, "cached_tokens", 0) or 0
            _record_usage(
                chunk.usage.prompt_tokens or 0,
                chunk.usage.completion_tokens or 0,
                model=model,
                cache_read_tokens=cache_read,
            )
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _default_embed_provider(name: str) -> Callable:
    if name == "openai":
        register_embed_provider("openai", _embed_openai)
        return _embed_openai
    if name == "voyage":
        register_embed_provider("voyage", _embed_voyage)
        return _embed_voyage
    raise ValueError(
        f"Unknown embed provider '{name}'. "
        f"Use register_embed_provider() or pick 'openai'/'voyage'."
    )


def _embed_openai(text: str, *, model: ModelConfig) -> list[float]:
    import openai

    kwargs: dict = {}
    api_key = os.environ.get(model.api_key_env or "OPENAI_API_KEY")
    if api_key:
        pass
    elif model.base_url:
        api_key = "ollama"
    if api_key:
        kwargs["api_key"] = api_key
    if model.base_url:
        kwargs["base_url"] = model.base_url

    client = openai.OpenAI(**kwargs)
    response = client.embeddings.create(
        model=model.model_id, input=[text]
    )
    return response.data[0].embedding


def _embed_voyage(text: str, *, model: ModelConfig) -> list[float]:
    import voyageai

    api_key = os.environ.get(model.api_key_env or "VOYAGE_API_KEY")
    client = voyageai.Client(api_key=api_key)
    result = client.embed([text], model=model.model_id)
    return result.embeddings[0]


async def async_ask(
    prompt: str,
    *,
    model: ModelConfig,
    system: str = "",
    max_tokens: int = 4096,
) -> str:
    """Send asynchronous text completion request to configuration model.

    Args:
        prompt: Instructions query.
        model: Configuration of target LLM.
        system: System instructions block.
        max_tokens: Maximum tokens permitted.

    Returns:
        The response content string.
    """
    return await asyncio.to_thread(
        ask, prompt, model=model, system=system, max_tokens=max_tokens
    )

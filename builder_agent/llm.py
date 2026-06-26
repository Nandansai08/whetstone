from __future__ import annotations

import json
import logging
import os
from typing import Callable, Generator

from builder_agent.config import ModelConfig

for _name in ("httpx", "httpcore", "openai", "anthropic"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.WARNING)
    _lg.propagate = False

_providers: dict[str, Callable] = {}
_stream_providers: dict[str, Callable] = {}
_embed_providers: dict[str, Callable] = {}
_budget = None


def set_budget(budget) -> None:
    global _budget
    _budget = budget


def get_budget():
    return _budget


def _record_usage(input_tokens: int, output_tokens: int) -> None:
    if _budget is not None:
        _budget.record(input_tokens, output_tokens)


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def extract_json(text: str) -> str:
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


def register_provider(name: str, fn: Callable) -> None:
    _providers[name] = fn


def register_stream_provider(name: str, fn: Callable) -> None:
    _stream_providers[name] = fn


def register_embed_provider(name: str, fn: Callable) -> None:
    _embed_providers[name] = fn


def embed(
    text: str,
    *,
    model: ModelConfig,
) -> list[float]:
    fn = _embed_providers.get(model.provider)
    if fn is None:
        fn = _default_embed_provider(model.provider)
    return fn(text, model=model)


def ask(
    prompt: str,
    *,
    model: ModelConfig,
    system: str = "",
    max_tokens: int = 4096,
) -> str:
    fn = _providers.get(model.provider)
    if fn is None:
        fn = _default_provider(model.provider)
    return fn(prompt, model=model, system=system, max_tokens=max_tokens)


def ask_stream(
    prompt: str,
    *,
    model: ModelConfig,
    system: str = "",
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    fn = _stream_providers.get(model.provider)
    if fn is None:
        if model.provider in ("anthropic", "openai"):
            fn = _default_stream_provider(model.provider)
        else:
            # Fallback to ask() and yield the entire response as a single chunk
            yield ask(prompt, model=model, system=system, max_tokens=max_tokens)
            return
    yield from fn(prompt, model=model, system=system, max_tokens=max_tokens)


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
        msg_kwargs["system"] = system
    response = client.messages.create(**msg_kwargs)
    if hasattr(response, "usage") and response.usage:
        _record_usage(
            response.usage.input_tokens,
            response.usage.output_tokens,
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
        _record_usage(
            response.usage.prompt_tokens or 0,
            response.usage.completion_tokens or 0,
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
        msg_kwargs["system"] = system

    with client.messages.stream(**msg_kwargs) as stream:
        for text in stream.text_stream:
            yield text
        message = stream.get_final_message()
        if message and hasattr(message, "usage") and message.usage:
            _record_usage(
                message.usage.input_tokens,
                message.usage.output_tokens,
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
            _record_usage(
                chunk.usage.prompt_tokens or 0,
                chunk.usage.completion_tokens or 0,
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

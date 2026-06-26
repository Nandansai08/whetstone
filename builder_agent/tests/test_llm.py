import asyncio
import os
from unittest.mock import MagicMock, patch

from builder_agent.config import ModelConfig
from builder_agent.llm import (
    ask,
    async_ask,
    embed,
    extract_json,
    register_embed_provider,
    register_provider,
    strip_fences,
)

ANTHROPIC_MODEL = ModelConfig("anthropic", "claude-sonnet-4-6")
OPENAI_MODEL = ModelConfig("openai", "gpt-4o")
CUSTOM_MODEL = ModelConfig(
    "openai", "llama3", api_key_env="LOCAL_KEY", base_url="http://localhost:11434/v1"
)


def _mock_anthropic_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def _mock_openai_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("builder_agent.llm._ask_anthropic")
def test_ask_dispatches_to_anthropic(mock_fn):
    mock_fn.return_value = "hello"
    result = ask("say hi", model=ANTHROPIC_MODEL)
    assert result == "hello"
    mock_fn.assert_called_once()


@patch("builder_agent.llm._ask_openai")
def test_ask_dispatches_to_openai(mock_fn):
    mock_fn.return_value = "hi there"
    result = ask("say hi", model=OPENAI_MODEL)
    assert result == "hi there"
    mock_fn.assert_called_once()


def test_register_custom_provider():
    calls = []

    def my_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append((prompt, model))
        return "custom response"

    register_provider("my_llm", my_provider)
    m = ModelConfig("my_llm", "my-model-v1")
    result = ask("test", model=m)
    assert result == "custom response"
    assert len(calls) == 1
    assert calls[0][1].model_id == "my-model-v1"


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
@patch("anthropic.Anthropic")
def test_anthropic_passes_model_id(mock_cls):
    client = MagicMock()
    client.messages.create.return_value = _mock_anthropic_response("ok")
    mock_cls.return_value = client

    from builder_agent.llm import _ask_anthropic
    _ask_anthropic("test", model=ANTHROPIC_MODEL)
    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-6"


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
@patch("anthropic.Anthropic")
def test_anthropic_passes_system(mock_cls):
    client = MagicMock()
    client.messages.create.return_value = _mock_anthropic_response("ok")
    mock_cls.return_value = client

    from builder_agent.llm import _ask_anthropic
    _ask_anthropic("test", model=ANTHROPIC_MODEL, system="be helpful")
    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["system"] == "be helpful"


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
@patch("anthropic.Anthropic")
def test_anthropic_omits_system_when_empty(mock_cls):
    client = MagicMock()
    client.messages.create.return_value = _mock_anthropic_response("ok")
    mock_cls.return_value = client

    from builder_agent.llm import _ask_anthropic
    _ask_anthropic("test", model=ANTHROPIC_MODEL)
    call_kwargs = client.messages.create.call_args[1]
    assert "system" not in call_kwargs


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
@patch("openai.OpenAI")
def test_openai_passes_model_id(mock_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_response("ok")
    mock_cls.return_value = client

    from builder_agent.llm import _ask_openai
    _ask_openai("test", model=OPENAI_MODEL)
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
@patch("openai.OpenAI")
def test_openai_uses_system_message(mock_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_response("ok")
    mock_cls.return_value = client

    from builder_agent.llm import _ask_openai
    _ask_openai("test", model=OPENAI_MODEL, system="be concise")
    call_kwargs = client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "be concise"}
    assert messages[1] == {"role": "user", "content": "test"}


@patch("openai.OpenAI")
def test_openai_custom_base_url(mock_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_response("ok")
    mock_cls.return_value = client

    from builder_agent.llm import _ask_openai
    _ask_openai("test", model=CUSTOM_MODEL)
    init_kwargs = mock_cls.call_args[1]
    assert init_kwargs["base_url"] == "http://localhost:11434/v1"


def test_unknown_provider_raises():
    m = ModelConfig("nonexistent", "whatever")
    try:
        ask("test", model=m)
        assert False, "Should have raised"
    except ValueError as e:
        assert "nonexistent" in str(e)


# --- embed() tests ---


def test_register_custom_embed_provider():
    calls = []

    def my_embedder(text, *, model):
        calls.append(text)
        return [0.1, 0.2, 0.3]

    register_embed_provider("my_embed", my_embedder)
    m = ModelConfig("my_embed", "embed-v1")
    result = embed("hello", model=m)
    assert result == [0.1, 0.2, 0.3]
    assert len(calls) == 1


@patch("openai.OpenAI")
def test_embed_openai_dispatches(mock_cls):
    emb_obj = MagicMock()
    emb_obj.embedding = [0.5, 0.6]
    resp = MagicMock()
    resp.data = [emb_obj]
    client = MagicMock()
    client.embeddings.create.return_value = resp
    mock_cls.return_value = client

    from builder_agent.llm import _embed_openai
    m = ModelConfig("openai", "text-embedding-3-small")
    result = _embed_openai("test text", model=m)
    assert result == [0.5, 0.6]
    call_kwargs = client.embeddings.create.call_args[1]
    assert call_kwargs["model"] == "text-embedding-3-small"


def test_unknown_embed_provider_raises():
    m = ModelConfig("nonexistent_embed", "whatever")
    try:
        embed("test", model=m)
        assert False, "Should have raised"
    except ValueError as e:
        assert "nonexistent_embed" in str(e)


# --- strip_fences() tests ---


def test_strip_fences_json():
    raw = '```json\n{"key": "value"}\n```'
    assert strip_fences(raw) == '{"key": "value"}'


def test_strip_fences_python():
    raw = '```python\ndef add(a, b):\n    return a + b\n```'
    assert strip_fences(raw) == 'def add(a, b):\n    return a + b'


def test_strip_fences_bare():
    raw = '```\n[1, 2, 3]\n```'
    assert strip_fences(raw) == '[1, 2, 3]'


def test_strip_fences_no_fences():
    raw = '{"score": 9, "issues": []}'
    assert strip_fences(raw) == raw


def test_strip_fences_whitespace():
    raw = '  \n```json\n{"a": 1}\n```\n  '
    assert strip_fences(raw) == '{"a": 1}'


# --- extract_json() tests ---


def test_extract_json_clean():
    assert extract_json('{"score": 9, "issues": []}') == '{"score": 9, "issues": []}'


def test_extract_json_trailing_text():
    raw = '{"score": 7, "issues": ["missing edge case"]}  Some extra commentary here.'
    result = extract_json(raw)
    import json
    data = json.loads(result)
    assert data["score"] == 7


def test_extract_json_leading_text():
    raw = 'Here is my analysis:\n{"score": 8, "issues": []}\nDone.'
    result = extract_json(raw)
    import json
    data = json.loads(result)
    assert data["score"] == 8


def test_extract_json_fenced_with_extra():
    raw = '```json\n{"score": 9, "issues": []}\n```\nAnd some more text.'
    result = extract_json(raw)
    import json
    data = json.loads(result)
    assert data["score"] == 9


def test_extract_json_array():
    raw = 'Tasks:\n[{"id": "t1", "description": "do stuff"}]\nEnd.'
    result = extract_json(raw)
    import json
    data = json.loads(result)
    assert len(data) == 1


def test_extract_json_nested_braces():
    raw = '{"a": {"b": 1}, "c": [1, 2]} extra'
    result = extract_json(raw)
    import json
    data = json.loads(result)
    assert data["a"]["b"] == 1


def test_extract_json_string_with_braces():
    raw = '{"msg": "use {x} and }"}  trailing'
    result = extract_json(raw)
    import json
    data = json.loads(result)
    assert data["msg"] == "use {x} and }"


def test_async_ask_fallback():
    custom_model = ModelConfig("custom_fallback", "some-model")

    with patch("builder_agent.llm.ask") as mock_ask:
        mock_ask.return_value = "sync fallback response"
        res = asyncio.run(
            async_ask("hello", model=custom_model, system="be quiet", max_tokens=100)
        )
        assert res == "sync fallback response"
        mock_ask.assert_called_once_with(
            "hello", model=custom_model, system="be quiet", max_tokens=100
        )


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
@patch("openai.AsyncOpenAI")
def test_async_ask_openai(mock_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "openai async response"

    async def mock_create(*args, **kwargs):
        return mock_response

    mock_client.chat.completions.create = mock_create
    mock_cls.return_value = mock_client

    res = asyncio.run(async_ask("hi", model=OPENAI_MODEL))
    assert res == "openai async response"


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
@patch("anthropic.AsyncAnthropic")
def test_async_ask_anthropic(mock_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "anthropic async response"

    async def mock_create(*args, **kwargs):
        return mock_response

    mock_client.messages.create = mock_create
    mock_cls.return_value = mock_client

    res = asyncio.run(async_ask("hi", model=ANTHROPIC_MODEL))
    assert res == "anthropic async response"

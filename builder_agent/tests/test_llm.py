import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from builder_agent import config
from builder_agent.config import ModelConfig
from builder_agent.llm import (
    _ask_anthropic,
    _ask_openai,
    _embed_openai,
    ask,
    ask_stream,
    async_ask,
    embed,
    extract_json,
    register_embed_provider,
    register_provider,
    register_stream_provider,
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

    _ask_anthropic("test", model=ANTHROPIC_MODEL)
    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-6"


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
@patch("anthropic.Anthropic")
def test_anthropic_passes_system(mock_cls):
    client = MagicMock()
    client.messages.create.return_value = _mock_anthropic_response("ok")
    mock_cls.return_value = client

    _ask_anthropic("test", model=ANTHROPIC_MODEL, system="be helpful")
    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs["system"] == "be helpful"


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
@patch("anthropic.Anthropic")
def test_anthropic_omits_system_when_empty(mock_cls):
    client = MagicMock()
    client.messages.create.return_value = _mock_anthropic_response("ok")
    mock_cls.return_value = client

    _ask_anthropic("test", model=ANTHROPIC_MODEL)
    call_kwargs = client.messages.create.call_args[1]
    assert "system" not in call_kwargs


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
@patch("openai.OpenAI")
def test_openai_passes_model_id(mock_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_response("ok")
    mock_cls.return_value = client

    _ask_openai("test", model=OPENAI_MODEL)
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
@patch("openai.OpenAI")
def test_openai_uses_system_message(mock_cls):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_response("ok")
    mock_cls.return_value = client

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
    data = json.loads(result)
    assert data["score"] == 7


def test_extract_json_leading_text():
    raw = 'Here is my analysis:\n{"score": 8, "issues": []}\nDone.'
    result = extract_json(raw)
    data = json.loads(result)
    assert data["score"] == 8


def test_extract_json_fenced_with_extra():
    raw = '```json\n{"score": 9, "issues": []}\n```\nAnd some more text.'
    result = extract_json(raw)
    data = json.loads(result)
    assert data["score"] == 9


def test_extract_json_array():
    raw = 'Tasks:\n[{"id": "t1", "description": "do stuff"}]\nEnd.'
    result = extract_json(raw)
    data = json.loads(result)
    assert len(data) == 1


def test_extract_json_nested_braces():
    raw = '{"a": {"b": 1}, "c": [1, 2]} extra'
    result = extract_json(raw)
    data = json.loads(result)
    assert data["a"]["b"] == 1


def test_extract_json_string_with_braces():
    raw = '{"msg": "use {x} and }"}  trailing'
    result = extract_json(raw)
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
@patch("builder_agent.llm._ask_stream_openai")
def test_ask_stream_dispatches_to_openai(mock_fn):
    mock_fn.return_value = (chunk for chunk in ["hi", " there"])
    result = list(ask_stream("say hi", model=OPENAI_MODEL))
    assert result == ["hi", " there"]
    mock_fn.assert_called_once()


@patch("builder_agent.llm._ask_stream_anthropic")
def test_ask_stream_dispatches_to_anthropic(mock_fn):
    mock_fn.return_value = (chunk for chunk in ["hello"])
    result = list(ask_stream("say hi", model=ANTHROPIC_MODEL))
    assert result == ["hello"]
    mock_fn.assert_called_once()


def test_register_custom_stream_provider():
    calls = []

    def my_stream_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append((prompt, model))
        yield "custom "
        yield "stream"

    register_stream_provider("my_stream_llm", my_stream_provider)
    m = ModelConfig("my_stream_llm", "my-model-v1")
    result = list(ask_stream("test", model=m))
    assert result == ["custom ", "stream"]
    assert len(calls) == 1
    assert calls[0][1].model_id == "my-model-v1"


def test_ask_stream_fallback_to_ask():
    calls = []

    def my_non_stream_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append((prompt, model))
        return "non-stream response"

    register_provider("my_non_stream_llm", my_non_stream_provider)
    m = ModelConfig("my_non_stream_llm", "my-model-v1")
    result = list(ask_stream("test", model=m))
    assert result == ["non-stream response"]
    assert len(calls) == 1
    assert calls[0][1].model_id == "my-model-v1"


@patch("builder_agent.llm.time.sleep")
def test_ask_retries_transient_failure_then_succeeds(mock_sleep):
    calls = []

    def flaky_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient")
        return "recovered"

    register_provider("flaky_llm", flaky_provider)
    m = ModelConfig("flaky_llm", "v1")
    result = ask("test", model=m)
    assert result == "recovered"
    assert len(calls) == 3
    assert mock_sleep.call_count == 2


@patch("builder_agent.llm.time.sleep")
def test_ask_gives_up_after_max_retries(mock_sleep):
    calls = []

    def always_fails(prompt, *, model, system="", max_tokens=4096):
        calls.append(1)
        raise ConnectionError("down")

    register_provider("dead_llm", always_fails)
    m = ModelConfig("dead_llm", "v1")
    with pytest.raises(ConnectionError):
        ask("test", model=m)
    assert len(calls) == config.MAX_RETRIES + 1


@patch("builder_agent.llm.time.sleep")
def test_ask_does_not_retry_config_errors(mock_sleep):
    calls = []

    def bad_key_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(1)
        raise RuntimeError("no api key")

    register_provider("bad_key_llm", bad_key_provider)
    m = ModelConfig("bad_key_llm", "v1")
    with pytest.raises(RuntimeError):
        ask("test", model=m)


@patch("time.sleep")
def test_retry_success_first_attempt(mock_sleep):
    calls = []

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        return "success"

    register_provider("test_retry_success", mock_provider)
    m = ModelConfig("test_retry_success", "model-v1")

    res = ask("hello", model=m)
    assert res == "success"
    assert len(calls) == 1
    mock_sleep.assert_not_called()


@patch("time.sleep")
def test_retry_once_then_succeeds(mock_sleep):
    calls = []
    import httpx

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        if len(calls) == 1:
            raise httpx.ConnectError("connection failed")
        return "success"

    register_provider("test_retry_once", mock_provider)
    m = ModelConfig("test_retry_once", "model-v1")

    progress_events = []
    def progress_callback(event, data):
        progress_events.append((event, data))

    from builder_agent.llm import set_progress_callback
    set_progress_callback(progress_callback)
    try:
        res = ask("hello", model=m)
        assert res == "success"
        assert len(calls) == 2
        mock_sleep.assert_called_once_with(1.0)
        assert len(progress_events) == 1
        assert progress_events[0][0] == "retry"
        assert progress_events[0][1]["attempt"] == 1
        assert progress_events[0][1]["delay"] == 1.0
        assert "connection failed" in progress_events[0][1]["error"]
    finally:
        set_progress_callback(None)


@patch("time.sleep")
def test_retry_multiple_times_then_succeeds(mock_sleep):
    calls = []
    import httpx

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        if len(calls) <= 2:
            raise httpx.ConnectError("connection failed")
        return "success"

    register_provider("test_retry_multiple", mock_provider)
    m = ModelConfig("test_retry_multiple", "model-v1")

    res = ask("hello", model=m)
    assert res == "success"
    assert len(calls) == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


@patch("time.sleep")
def test_retry_exhausted(mock_sleep):
    calls = []
    import httpx

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        raise httpx.ConnectError("connection failed")

    register_provider("test_retry_exhausted", mock_provider)
    m = ModelConfig("test_retry_exhausted", "model-v1")

    try:
        ask("hello", model=m)
        assert False, "Should have raised exception"
    except httpx.ConnectError as e:
        assert "connection failed" in str(e)

    assert len(calls) == 4  # Initial try + 3 retries
    assert mock_sleep.call_count == 3
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(4.0)


@patch("time.sleep")
def test_retry_429_openai(mock_sleep):
    calls = []
    import httpx
    import openai

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        if len(calls) == 1:
            req = httpx.Request("POST", "https://api.openai.com")
            resp = httpx.Response(status_code=429, request=req)
            raise openai.APIStatusError("Rate Limit", response=resp, body=None)
        return "success"

    register_provider("test_retry_429_openai", mock_provider)
    m = ModelConfig("test_retry_429_openai", "model-v1")

    res = ask("hello", model=m)
    assert res == "success"
    assert len(calls) == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("time.sleep")
def test_retry_500_anthropic(mock_sleep):
    calls = []
    import anthropic
    import httpx

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        if len(calls) == 1:
            req = httpx.Request("POST", "https://api.anthropic.com")
            resp = httpx.Response(status_code=500, request=req)
            raise anthropic.APIStatusError(
                "Internal Server Error", response=resp, body=None
            )
        return "success"

    register_provider("test_retry_500_anthropic", mock_provider)
    m = ModelConfig("test_retry_500_anthropic", "model-v1")

    res = ask("hello", model=m)
    assert res == "success"
    assert len(calls) == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("time.sleep")
def test_retry_503_generic(mock_sleep):
    calls = []
    import httpx

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        if len(calls) == 1:
            req = httpx.Request("POST", "https://api.generic.com")
            resp = httpx.Response(status_code=503, request=req)
            raise httpx.HTTPStatusError(
                "Service Unavailable", request=resp.request, response=resp
            )
        return "success"

    register_provider("test_retry_503_generic", mock_provider)
    m = ModelConfig("test_retry_503_generic", "model-v1")

    res = ask("hello", model=m)
    assert res == "success"
    assert len(calls) == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("time.sleep")
def test_no_retry_on_bad_request_or_auth(mock_sleep):
    calls = []
    import httpx
    import openai

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        req = httpx.Request("POST", "https://api.openai.com")
        resp = httpx.Response(status_code=401, request=req)
        raise openai.AuthenticationError(
            "Invalid API Key", response=resp, body=None
        )

    register_provider("test_no_retry_401", mock_provider)
    m = ModelConfig("test_no_retry_401", "model-v1")

    try:
        ask("hello", model=m)
        assert False, "Should have raised"
    except openai.AuthenticationError:
        pass

    assert len(calls) == 1
    mock_sleep.assert_not_called()


@patch("time.sleep")
def test_no_retry_on_validation_error(mock_sleep):
    calls = []
    import httpx
    import openai

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        req = httpx.Request("POST", "https://api.openai.com")
        resp = httpx.Response(status_code=400, request=req)
        raise openai.BadRequestError(
            "Validation failed", response=resp, body=None
        )

    register_provider("test_no_retry_400", mock_provider)
    m = ModelConfig("test_no_retry_400", "model-v1")

    try:
        ask("hello", model=m)
        assert False, "Should have raised"
    except openai.BadRequestError:
        pass

    assert len(calls) == 1
    mock_sleep.assert_not_called()


@patch("time.sleep")
def test_no_retry_on_keyboard_interrupt(mock_sleep):
    calls = []

    def mock_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        raise KeyboardInterrupt()

    register_provider("test_keyboard_interrupt", mock_provider)
    m = ModelConfig("test_keyboard_interrupt", "model-v1")

    try:
        ask("hello", model=m)
        assert False, "Should have raised"
    except KeyboardInterrupt:
        pass

    assert len(calls) == 1
    mock_sleep.assert_not_called()


@patch("time.sleep")
def test_ask_stream_retry_before_start(mock_sleep):
    calls = []
    import httpx

    def mock_stream_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        if len(calls) == 1:
            raise httpx.ConnectError("connection failed")
        yield "chunk1"
        yield "chunk2"

    register_stream_provider("test_stream_retry", mock_stream_provider)
    m = ModelConfig("test_stream_retry", "model-v1")

    res = list(ask_stream("hello", model=m))
    assert res == ["chunk1", "chunk2"]
    assert len(calls) == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("time.sleep")
def test_ask_stream_no_retry_mid_stream(mock_sleep):
    calls = []
    import httpx

    def mock_stream_provider(prompt, *, model, system="", max_tokens=4096):
        calls.append(prompt)
        yield "chunk1"
        raise httpx.ConnectError("midstream failed")

    register_stream_provider("test_stream_mid", mock_stream_provider)
    m = ModelConfig("test_stream_mid", "model-v1")

    res = []
    try:
        for chunk in ask_stream("hello", model=m):
            res.append(chunk)
        assert False, "Should have failed mid-stream"
    except httpx.ConnectError as e:
        assert "midstream failed" in str(e)

    assert res == ["chunk1"]
    assert len(calls) == 1
    mock_sleep.assert_not_called()

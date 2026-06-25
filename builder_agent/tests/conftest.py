import pytest


@pytest.fixture(autouse=True)
def mock_generate_ask_stream_fallback():
    from unittest.mock import Mock

    from builder_agent import generate

    original_ask_stream = generate.ask_stream

    def wrapped_ask_stream(prompt, *, model, system="", max_tokens=4096):
        if isinstance(generate.ask, Mock):
            # If ask is mocked (e.g. by test_orchestrate.py),
            # call it and yield the result
            res = generate.ask(
                prompt, model=model, system=system, max_tokens=max_tokens
            )
            yield res
        else:
            yield from original_ask_stream(
                prompt, model=model, system=system, max_tokens=max_tokens
            )

    generate.ask_stream = wrapped_ask_stream
    yield
    generate.ask_stream = original_ask_stream

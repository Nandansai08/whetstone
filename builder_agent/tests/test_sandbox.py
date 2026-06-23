from builder_agent.sandbox import run_code


def test_good_code_passes():
    passed, output = run_code("print('hello')")
    assert passed is True
    assert "hello" in output


def test_bad_code_fails():
    passed, output = run_code("raise ValueError('boom')")
    assert passed is False
    assert "ValueError" in output


def test_syntax_error_fails():
    passed, output = run_code("def f(\n")
    assert passed is False


def test_timeout_kills_infinite_loop():
    passed, output = run_code("while True: pass", timeout=2)
    assert passed is False
    assert "Timeout" in output


def test_exit_code_nonzero_fails():
    passed, output = run_code("import sys; sys.exit(1)")
    assert passed is False


def test_multiline_output():
    code = "print('line1')\nprint('line2')"
    passed, output = run_code(code)
    assert passed is True
    assert "line1" in output
    assert "line2" in output

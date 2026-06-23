from builder_agent.budget import TokenBudget


def test_budget_starts_at_zero():
    b = TokenBudget(limit=1000)
    assert b.total == 0
    assert b.input_tokens == 0
    assert b.output_tokens == 0
    assert not b.exceeded()


def test_budget_accumulates():
    b = TokenBudget(limit=1000)
    b.record(100, 50)
    b.record(200, 75)
    assert b.input_tokens == 300
    assert b.output_tokens == 125
    assert b.total == 425
    assert not b.exceeded()


def test_budget_exceeded():
    b = TokenBudget(limit=100)
    b.record(60, 50)
    assert b.exceeded()


def test_budget_zero_limit_never_exceeded():
    b = TokenBudget(limit=0)
    b.record(999999, 999999)
    assert not b.exceeded()


def test_budget_usage_dict():
    b = TokenBudget(limit=500)
    b.record(100, 200)
    u = b.usage()
    assert u == {
        "input_tokens": 100,
        "output_tokens": 200,
        "total_tokens": 300,
        "limit": 500,
    }

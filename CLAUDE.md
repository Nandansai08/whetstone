# Whetstone — Builder Agent

A meta-agent: user says "I want to build X" → it clarifies, plans into
sub-tasks, builds each with a worker LLM, verifies with executable tests + a
cross-model judge, refines on failure, and stores what worked in memory.

## Stack
- Python 3.11+, dataclasses. Provider SDKs (`openai`, optionally `anthropic`) as deps.
- Memory: SQLite + pluggable embedder (sentence-transformers, TF-IDF, or LLM-based).

## Rules (always)
- All LLM calls go through `llm.py`. Never call any provider SDK directly from
  any other module.
- `llm.py` is provider-agnostic. Models configured via `ModelConfig(provider, model_id)`.
  Built-in providers: `anthropic`, `openai`. Custom providers via `register_provider()`.
- No live API calls in the test suite — mock the `llm.py` wrapper.
- Every component has unit tests; a feature isn't done until its tests pass.
- The verifier requires BOTH: independently-generated tests pass AND judge
  score ≥ threshold. `JUDGE_MODEL` must differ from `WORKER_MODEL`.
- Always return the best-so-far attempt; never silently ship a failed verify.

## Security
- `sandbox.py` runs model-generated code via subprocess + timeout. That is NOT
  isolation. Do not point it at untrusted input.

## Commands
- Test: `pytest`
- Lint: `ruff check builder_agent/`
- Run: `python -m builder_agent` or `whetstone`

## Architecture
- Full spec and architecture: `docs/ARCHITECTURE.md`

# Whetstone

AI-powered code builder. Describe what you want, get working code.

Whetstone takes a natural language request, clarifies it into a spec, plans subtasks, generates code with a worker LLM, verifies it with executable tests + a cross-model judge, refines on failure, and remembers what worked for next time.

```
                 ┌──────────┐
   request  ───▶ │ CLARIFY  │  request → Spec
                 └────┬─────┘
                      ▼
                 ┌──────────┐
                 │   PLAN   │◀── memory (past builds)
                 └────┬─────┘
                      ▼
        for each SubTask:
        ┌───────────────────────────┐
        │  GENERATE → SELF-CRITIQUE │◀── memory hints
        │       ▲          │        │
        │   feedback    VERIFY      │
        │       └──── fail ◀──┘     │
        └───────────┬───────────────┘
                    ▼
                 INTEGRATE → FINAL VERIFY → done
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Set your API key (or create a .env file — see .env.example)
export OPENROUTER_API_KEY=sk-or-v1-...

# Launch
whetstone
# or
python -m builder_agent
```

Then type what you want to build:

```
  ❯ build a CSV parser with custom delimiters

  ──────────────────────────────────────────────────
  Build #1
  ──────────────────────────────────────────────────
  ✓ Clarified (4.2s)
    Build a CSV parser that handles quoted fields and custom delimiters
  ✓ Plan: 1 subtask (2.1s)
    t1 implement csv parser

  [1/1] t1 implement csv parser
  ✓ iter 1 score 9/10 (8.3s)

  ──────────────────────────────────────────────────
  ● BUILD PASSED
    Score [████████████████████░░░░] 9/10
    Stats  1 subtasks · 1 iters
    Tokens 4,231 / 200,000
```

## Features

- **Provider-agnostic** — works with OpenRouter, OpenAI, Anthropic, Ollama, or any OpenAI-compatible API
- **Cross-model verification** — worker and judge are different models to avoid blind spots
- **Test-first verification** — generates executable tests from acceptance criteria before checking
- **Self-critique** — worker reviews its own code before the judge sees it
- **Plateau detection** — stops wasting iterations when scores stop improving, escalates to a stronger model
- **Token budget** — tracks cumulative usage, aborts gracefully when exceeded
- **Memory** — stores what failed and what fixed it; retrieves similar past builds as hints
- **Interactive CLI** — spinner progress, score bars, line-numbered code output

## Configuration

Edit `builder_agent/config.py` to change models:

```python
from builder_agent.config import ModelConfig

# OpenRouter (default)
_OPENROUTER = "https://openrouter.ai/api/v1"
_OR_KEY = "OPENROUTER_API_KEY"

WORKER_MODEL = ModelConfig("openai", "meta-llama/llama-4-scout",
                           api_key_env=_OR_KEY, base_url=_OPENROUTER)
JUDGE_MODEL  = ModelConfig("openai", "google/gemini-2.5-flash-preview",
                           api_key_env=_OR_KEY, base_url=_OPENROUTER)
```

### Using Ollama (local, no API key)

```python
_OLLAMA = "http://localhost:11434/v1"

WORKER_MODEL = ModelConfig("openai", "gemma3:12b", base_url=_OLLAMA)
JUDGE_MODEL  = ModelConfig("openai", "qwen2.5:14b", base_url=_OLLAMA)
```

### Using OpenAI directly

```python
WORKER_MODEL = ModelConfig("openai", "gpt-4o-mini", api_key_env="OPENAI_API_KEY")
JUDGE_MODEL  = ModelConfig("openai", "gpt-4o", api_key_env="OPENAI_API_KEY")
```

## CLI Usage

### Interactive mode (default)

```bash
whetstone          # launches REPL
```

Commands inside the REPL:

| Command | Description |
|---------|-------------|
| `<request>` | Build something |
| `/config` | Show model configuration |
| `/memory` | List stored memory records |
| `/memory show <id>` | Show a specific record |
| `/memory clear` | Clear all records |
| `/help` | Show help |
| `/quit` | Exit |

### One-shot mode

```bash
whetstone build "Build a function add(a, b) that returns a + b" --non-interactive
whetstone build "Build a CSV parser" --output parser.py
whetstone build "Build a binary search" --json
```

### Memory management

```bash
whetstone memory list
whetstone memory show 1
whetstone memory clear --yes
```

## Architecture

```
builder_agent/
  config.py       — models, thresholds, budgets
  llm.py          — provider-agnostic ask() and embed()
  schemas.py      — dataclasses (Spec, SubTask, Plan, Verdict, Attempt, MemoryRecord)
  sandbox.py      — subprocess execution with timeout
  clarify.py      — request → Spec
  plan.py         — Spec → Plan (ordered subtasks)
  generate.py     — SubTask → code (with self-critique)
  verify.py       — code → Verdict (tests + cross-model judge)
  memory.py       — SQLite store/retrieve with embeddings
  embedders.py    — pluggable: sentence-transformers, TF-IDF, LLM-based
  integrate.py    — combine subtask outputs into final artifact
  orchestrate.py  — the state machine wiring it all together
  budget.py       — token budget tracking
  cli.py          — interactive REPL + one-shot CLI
  tests/          — 127 tests, all LLM calls mocked
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check builder_agent/
```

## How It Works

1. **Clarify** — LLM converts your request into a structured spec with acceptance criteria
2. **Plan** — Decomposes spec into subtasks with dependency ordering (simple tasks → 1 subtask)
3. **Per subtask:**
   - **Generate** code from the spec + any feedback from prior iterations
   - **Self-critique** — worker reviews and improves its own draft
   - **Verify** — generate tests from acceptance criteria, run in sandbox, then cross-model judge scores 0-10
   - If failed: feed issues as feedback, try again (up to MAX_ITERATIONS)
   - If plateau detected: escalate to stronger model
   - If budget exceeded: stop and return best-so-far
4. **Integrate** — assemble subtask outputs in dependency order
5. **Final verify** — run the whole artifact through verification one more time

## Security

The sandbox executes model-generated code via `subprocess` with a timeout. **This is NOT isolation.** Do not use with untrusted input. For production use, swap in a container with no network and resource caps.

## License

MIT

<p align="center">
  <pre>
 __        ___          _       _
 \ \      / / |__   ___| |_ ___| |_ ___  _ __   ___
  \ \ /\ / /| '_ \ / _ \ __/ __| __/ _ \| '_ \ / _ \
   \ V  V / | | | |  __/ |_\__ \ || (_) | | | |  __/
    \_/\_/  |_| |_|\___|\__|___/\__\___/|_| |_|\___|
  </pre>
</p>

<p align="center">
  <strong>AI-powered code builder — describe what you want, get working code.</strong>
</p>

<p align="center">
  <a href="https://github.com/Nandansai08/whetstone/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Nandansai08/whetstone/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
  <a href="https://github.com/Nandansai08/whetstone/releases"><img src="https://img.shields.io/github/v/release/Nandansai08/whetstone?style=flat-square" alt="Release"></a>
  <a href="https://github.com/Nandansai08/whetstone/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Nandansai08/whetstone?style=flat-square" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python"></a>
  <a href="https://github.com/Nandansai08/whetstone/issues"><img src="https://img.shields.io/github/issues/Nandansai08/whetstone?style=flat-square" alt="Issues"></a>
</p>

---

Whetstone takes a natural language request, clarifies it into a spec, plans subtasks, generates code with a worker LLM, verifies it with executable tests + a cross-model judge, refines on failure, and remembers what worked for next time.

## How It Works

```
  "Build a CSV parser"
         │
         ▼
  ┌─────────────┐
  │   CLARIFY   │  natural language → structured Spec
  └──────┬──────┘  (acceptance criteria, assumptions, output type)
         │
         ▼
  ┌─────────────┐
  │    PLAN     │◄──── memory: similar past builds
  └──────┬──────┘      (what worked, what didn't)
         │
         │  Plan: ordered subtasks with dependencies
         ▼
  ┌─────────────────────────────────────────┐
  │  for each subtask:                      │
  │                                         │
  │   GENERATE ──► SELF-CRITIQUE            │
  │       ▲              │                  │
  │       │              ▼                  │
  │    feedback      VERIFY                 │
  │       │      (sandbox tests + judge)    │
  │       │              │                  │
  │       └── fail ◄─────┘                  │
  │                                         │
  │   plateau? → escalate to stronger model │
  │   budget?  → stop, return best-so-far   │
  └───────────────┬─────────────────────────┘
                  │  pass
                  ▼
           ┌─────────────┐
           │  INTEGRATE   │  assemble subtask outputs
           └──────┬──────┘
                  ▼
           ┌─────────────┐
           │ FINAL VERIFY │  verify the whole artifact
           └──────┬──────┘
                  ▼
              ✓ done
```

**Key insight:** the worker and judge are *different models* — their blind spots don't overlap, so the verification is genuinely independent.

## Demo

```
 __        ___          _       _
 \ \      / / |__   ___| |_ ___| |_ ___  _ __   ___
  \ \ /\ / /| '_ \ / _ \ __/ __| __/ _ \| '_ \ / _ \
   \ V  V / | | | |  __/ |_\__ \ || (_) | | | |  __/
    \_/\_/  |_| |_|\___|\__|___/\__\___/|_| |_|\___|
  Builder Agent v0.1.0  ·  meta-llama/llama-4-scout

  Type what you want to build, or /help for commands.

  ❯ build a CSV parser with custom delimiters

  ──────────────────────────────────────────────────
  Build #1
  ──────────────────────────────────────────────────
  ✓ Clarified (4.2s)
    Build a CSV parser that handles quoted fields and custom delimiters
  ✓ Plan: 1 subtask (2.1s)
    t1 implement csv parser

  [1/1] t1 implement csv parser
  ⠹ Generating iter 1 3s
  ✓ iter 1 score 9/10 (8.3s)

  ──────────────────────────────────────────────────
  ● BUILD PASSED
    Score [████████████████████░░░░] 9/10
    Stats  1 subtasks · 1 iters
    Tokens 4,231 / 200,000

  Output 28 lines
  ┌────────────────────────────────────────────────────────────
  1 │ import csv
  2 │ from io import StringIO
  3 │
  4 │ def parse_csv(text, delimiter=',', quotechar='"'):
  5 │     """Parse CSV text with custom delimiter and quote char."""
  6 │     reader = csv.reader(
  7 │         StringIO(text),
  8 │         delimiter=delimiter,
  9 │         quotechar=quotechar
 10 │     )
 11 │     ...
  └────────────────────────────────────────────────────────────
    Time   14.6s

  ❯ /config

  Models
    worker       meta-llama/llama-4-scout (openai)
    judge        google/gemini-2.5-flash-preview (openai)
    planner      meta-llama/llama-4-scout (openai)
    escalation   google/gemini-2.5-flash-preview (openai)
    endpoint     https://openrouter.ai/api/v1

  Settings
    threshold    8/10
    max_iter     4
    patience     2
    budget       200,000 tokens
    memory       ./builder_memory.db

  ❯ /memory

  ◇   1 Build a CSV parser with custom delimiters     2025-06-23 14:32

  ❯ /quit
  Goodbye.
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/Nandansai08/whetstone.git
cd whetstone
pip install -e ".[dev]"

# Set your API key
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env

# Launch
whetstone
```

> **No API key?** Use [Ollama](https://ollama.com) for free local models — see [Configuration](#configuration) below.

## Features

| Feature | Description |
|---------|-------------|
| **Provider-agnostic** | OpenRouter, OpenAI, Anthropic, Ollama, or any OpenAI-compatible API |
| **Cross-model verification** | Worker ≠ judge — independent blind spots catch more bugs |
| **Test-first** | Generates executable tests from acceptance criteria before verifying |
| **Self-critique** | Worker reviews its own code before the judge sees it |
| **Plateau detection** | Stops wasting iterations when stuck, escalates to stronger model |
| **Token budget** | Tracks cumulative usage, aborts gracefully when exceeded |
| **Memory** | Remembers what failed and what fixed it across builds |
| **Smart planning** | Simple tasks → 1 subtask. Complex tasks → dependency-ordered plan |
| **Interactive CLI** | Spinners, score bars, line-numbered code, timing per stage |
| **One-shot mode** | `whetstone build "..." --json` for scripting and CI |

## Configuration

Instead of editing `builder_agent/config.py` directly, you can configure Whetstone using TOML configuration files.

### Configuration Search Order
Whetstone loads configuration settings in the following order of precedence (highest to lowest):
1. **CLI arguments** (e.g., `--max-iterations`, `--token-budget`).
2. **Project-local config**: `.whetstone.toml` in the current project directory.
3. **Global user config**: `~/.config/whetstone/config.toml` in your home directory.
4. **Defaults**: Core defaults fallback defined in `builder_agent/config.py`.

### Generating a Default Config File
To initialize a default configuration file in your current project root, run:
```bash
whetstone init
```
This generates a `.whetstone.toml` file with standard configurations and documented providers. If the file already exists, the command will exit safely without modifying it.

### Example Configuration (`.whetstone.toml`)
```toml
# General settings
max_iterations = 4
score_threshold = 8
plateau_patience = 2
exec_timeout = 10
token_budget = 200000
embedder = "tfidf"
max_subtasks = 5
max_retries = 3
retry_delay = 1.0

# Models config
[models.worker]
provider = "openai"
model_id = "meta-llama/llama-4-scout"
api_key_env = "OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"

[models.judge]
provider = "openai"
model_id = "google/gemini-2.5-flash-preview"
api_key_env = "OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"

# Memory config
[memory]
db_path = "./builder_memory.db"
top_k = 3
min_similarity = 0.4

# Sandbox settings
[sandbox]
backend = "subprocess"
engine = "docker"
image = "python:3.11-slim"
memory_limit = "256m"
cpu_limit = 1.0
network_access = false
```

## CLI Reference

### Interactive mode (default)

```bash
whetstone          # launches REPL
```

| Command | Description |
|---------|-------------|
| `<any text>` | Build something |
| `/config` | Show model configuration |
| `/memory` | List stored memory records |
| `/memory show <id>` | Show a specific record |
| `/memory clear` | Clear all records |
| `/help` | Show available commands |
| `/quit` | Exit |

### One-shot mode

```bash
whetstone build "Build a binary search" --non-interactive
whetstone build "Build a CSV parser" --output parser.py
whetstone build "Build add(a,b)" --json              # structured output
whetstone build "Build a REST client" --max-iterations 6
whetstone build "Build a sort" --token-budget 50000
whetstone build "Build a cache" --no-memory           # skip memory read/write
```

### Memory management

```bash
whetstone memory list
whetstone memory list --type plan        # only plan records
whetstone memory list --type subtask     # only subtask records
whetstone memory show 1                  # show record details
whetstone memory clear --yes             # delete all records
```

## Architecture

```
builder_agent/
├── config.py        models, thresholds, budgets, paths
├── llm.py           provider-agnostic ask() and embed() — the ONLY API surface
├── schemas.py       Spec, SubTask, Plan, Verdict, Attempt, MemoryRecord
├── clarify.py       request → Spec (acceptance criteria, assumptions)
├── plan.py          Spec → Plan (topo-sorted subtasks, smart cap)
├── generate.py      SubTask → code (with self-critique)
├── verify.py        code → Verdict (sandbox tests + cross-model judge)
├── sandbox.py       subprocess execution with timeout
├── memory.py        SQLite store/retrieve with cosine similarity
├── embedders.py     pluggable: sentence-transformers, TF-IDF, LLM-based
├── integrate.py     combine subtask outputs, dedupe imports, ast.parse
├── orchestrate.py   state machine with progress callbacks
├── budget.py        thread-safe token budget tracking
├── cli.py           interactive REPL + one-shot CLI + spinner UI
├── __main__.py      python -m builder_agent entry point
└── tests/           127 tests, all LLM calls mocked
```

### Verification pipeline

The verifier is the core of Whetstone's reliability. Three independent strengthenings:

1. **Test-first** — `make_tests()` derives executable tests from acceptance criteria *before* generation. Run in sandbox = objective pass/fail.
2. **Cross-model judge** — `JUDGE_MODEL ≠ WORKER_MODEL` so their blind spots don't overlap (a model grading itself is too lenient).
3. **Self-critique** — worker revises once before verification; consistent quality bump for one extra call.

`passed` requires **both** tests to pass **and** judge score ≥ threshold.

### Control loop (per subtask)

```
attempts = 0; best = None
while attempts < MAX_ITERATIONS:
    code     = generate(subtask, feedback, memory_hints)
    code     = self_critique(code)
    verdict  = verify(subtask, code)     # tests + judge
    track best by score
    if passed:  break
    if plateau: escalate to stronger model (once)
    if budget:  break
    feedback = verdict.issues
    attempts += 1
return best   ← always return highest-scoring attempt
```

## Development

```bash
# Install
pip install -e ".[dev]"

# Test (127 tests, ~10s, no API calls)
pytest

# Lint
ruff check builder_agent/

# Run
python -m builder_agent
```

### Adding a custom provider

```python
from builder_agent.llm import register_provider

def my_provider(prompt, *, model, system="", max_tokens=4096):
    # call your LLM here
    return response_text

register_provider("my_llm", my_provider)
```

Then use it in config:

```python
WORKER_MODEL = ModelConfig("my_llm", "my-model-id")
```

## Security

The sandbox executes model-generated code via `subprocess` with a timeout. **This is NOT isolation.** Do not use with untrusted input. For production use, swap in a container with no network and resource caps. See [#3](https://github.com/Nandansai08/whetstone/issues/3).

## Contributing

See [open issues](https://github.com/Nandansai08/whetstone/issues) — issues labeled [`good first issue`](https://github.com/Nandansai08/whetstone/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are great starting points.

```bash
git clone https://github.com/Nandansai08/whetstone.git
cd whetstone
pip install -e ".[dev]"
pytest                    # make sure everything passes
# make your changes
ruff check builder_agent/ # lint before committing
```

## License

[MIT](LICENSE)

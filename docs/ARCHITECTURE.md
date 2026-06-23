# Builder Agent — Architecture & Build Spec (for Claude Code)

> **How to use this doc:** Build in milestone order (M1 → M5). Each milestone is
> independently testable — write its unit tests and make them pass before moving
> on. Don't build the whole thing at once. Modules that touch the LLM should be
> behind a thin wrapper so they can be mocked in tests (no live API calls in the
> test suite). Use Python 3.11+, the `anthropic` SDK, and dataclasses.

A meta-agent: user says "I want to build X" → the agent clarifies it, **plans**
it into sub-tasks, builds each with a worker, **verifies** with executable
tests + a cross-model judge, refines on failure, and **remembers** what worked
so it improves across runs.

```
                 ┌──────────┐
   request  ───▶ │ CLARIFY  │  request -> Spec (criteria + assumptions)
                 └────┬─────┘
                      ▼
                 ┌──────────┐   reads similar past builds
                 │  PLAN    │◀──────────────┐
                 └────┬─────┘                │
                      │ Plan (ordered SubTasks)
                      ▼                       │
        for each SubTask:                     │
        ┌───────────────────────────┐        │
        │  GENERATE ─▶ SELF-CRITIQUE │        │   ┌──────────┐
        │       ▲          │         │◀───────┼───│  MEMORY  │
        │       │          ▼         │ hints  │   │ (SQLite +│
        │   feedback    VERIFY       │        │   │ embeds)  │
        │       │   (tests + judge)  │        │   └────▲─────┘
        │       └──── fail ◀──┘      │             writes │
        └───────────┬───────────────┘                    │
                    │ pass / plateau / budget             │
                    ▼                                      │
                 ┌──────────┐                              │
                 │INTEGRATE │ assemble sub-outputs ────────┘
                 └────┬─────┘
                      ▼
                 ┌──────────┐
                 │FINAL     │ verify the whole, then NOTIFY
                 │VERIFY    │
                 └──────────┘
```

---

## Project layout

```
builder_agent/
  config.py          # models, thresholds, budgets, paths
  llm.py             # Anthropic wrapper: ask(), embed(); the ONLY API surface
  schemas.py         # dataclasses (contracts below)
  sandbox.py         # safe-ish subprocess execution with timeout
  clarify.py         # request -> Spec
  plan.py            # Spec (+memory) -> Plan
  generate.py        # SubTask (+feedback, +memory hints) -> code; self_critique()
  verify.py          # test-first tests + execute + cross-model judge -> Verdict
  memory.py          # store/retrieve past builds (pluggable embedder)
  integrate.py       # combine sub-task outputs -> final artifact
  orchestrate.py     # the state machine wiring it all together
  cli.py             # entrypoint: python -m builder_agent "build a..."
  tests/             # one test module per component, LLM mocked
```

---

## Data contracts (`schemas.py`)

```python
@dataclass
class Spec:
    request: str
    description: str                 # concrete spec prose
    acceptance_criteria: list[str]   # objective, checkable statements
    assumptions: list[str]
    output_type: str                 # "python_module" | "sql" | "pipeline"

@dataclass
class SubTask:
    id: str
    description: str
    acceptance_criteria: list[str]
    depends_on: list[str]            # ids of prerequisite subtasks

@dataclass
class Plan:
    subtasks: list[SubTask]          # topologically ordered

@dataclass
class Verdict:
    passed: bool                     # tests_passed AND score >= threshold
    score: int                       # 0-10 from judge
    tests_passed: bool
    issues: list[str]                # concrete, become next-round feedback
    exec_output: str

@dataclass
class Attempt:
    iteration: int
    code: str
    verdict: Verdict

@dataclass
class MemoryRecord:
    request: str
    output_type: str
    subtask_desc: str
    failures: list[str]              # what went wrong
    fix_summary: str                 # what resolved it
    final_code: str
    embedding: list[float]           # of (request + subtask_desc)
```

---

## Component contracts

```python
# clarify.py
def clarify(request: str, *, interactive: bool = True) -> Spec
    # one round of <=3 high-value questions; sensible defaults otherwise.
    # Must populate acceptance_criteria — they drive verification downstream.

# plan.py
def plan(spec: Spec, memory: "Memory") -> Plan
    # decompose into ordered subtasks. Single-step tasks => a one-item plan.
    # Inject similar past builds from memory as planning context.

# generate.py
def generate(subtask: SubTask, spec: Spec,
             feedback: str | None = None,
             memory_hints: list[MemoryRecord] | None = None) -> str
    # worker. Returns code for ONE subtask. feedback carries exact prior
    # failures; memory_hints carries fixes that worked on similar tasks.

def self_critique(code: str, subtask: SubTask) -> str
    # worker reviews + improves its own draft BEFORE the verifier sees it.

# verify.py
def make_tests(subtask: SubTask, code: str) -> str   # test-first: from criteria
def verify(subtask: SubTask, code: str) -> Verdict
    # 1) run independently-generated tests in the sandbox (objective)
    # 2) judge with JUDGE_MODEL (!= worker model) against the criteria rubric
    # passed = tests_passed AND score >= SCORE_THRESHOLD

# sandbox.py
def run_code(code: str, timeout: int) -> tuple[bool, str]   # (passed, output)

# memory.py  (class Memory)
def store(self, record: MemoryRecord) -> None
def retrieve(self, query: str, k: int) -> list[MemoryRecord]  # cosine top-k

# integrate.py
def integrate(spec: Spec, outputs: dict[str, str]) -> str
    # assemble per-subtask code into one artifact (modules/files).

# orchestrate.py
def orchestrate(request: str) -> dict   # the state machine; returns result
```

---

## Control loop (per subtask) — with the guards that matter

```
attempt = 0; best = None; feedback = None
hints = memory.retrieve(subtask.description, k=MEMORY_TOP_K)
while attempt < MAX_ITERATIONS:
    code   = generate(subtask, spec, feedback, hints)
    code   = self_critique(code, subtask)
    v      = verify(subtask, code)
    track best by score
    if v.passed: break
    if plateaued(history, patience=PLATEAU_PATIENCE):  # score not improving
        escalate()        # switch to stronger model OR ask the user one Q
    if over_budget():     # cumulative token budget
        break
    feedback = "\n".join(v.issues)   # objective failures first
    attempt += 1
return best
```

Guards, explicitly:
- **MAX_ITERATIONS** — hard stop per subtask.
- **Plateau detection** — if score doesn't improve by ≥1 over `PLATEAU_PATIENCE`
  iterations, stop refining blindly: escalate to a stronger worker model, or
  surface one targeted question to the user. Don't burn calls on a stuck loop.
- **Token budget** — track cumulative usage; abort gracefully when exceeded.
- **Best-so-far** — always return the highest-scoring attempt, never assume the
  last one is best. On a non-pass exit, flag it honestly.

---

## Verifier design (the crux)

Three independent strengthenings, all cheap:
1. **Test-first** — `make_tests()` derives executable tests from the subtask's
   `acceptance_criteria` *before* generation, so the worker aims at a fixed
   target. Run them in the sandbox = objective pass/fail.
2. **Cross-model judge** — `JUDGE_MODEL` differs from `WORKER_MODEL` so their
   blind spots don't overlap (a model grading itself is too lenient).
3. **Self-critique** — worker revises once before verification; consistent
   quality bump for one extra call.

`passed` requires **both** the tests to pass **and** the judge ≥ threshold.
Objective failure short-circuits the judge.

---

## Memory subsystem

The thing that makes it improve *across* tasks, not just within one.

- **Storage:** SQLite table of `MemoryRecord` rows; embeddings stored as JSON.
- **Embedder (pluggable):** default to a local `sentence-transformers`
  (`all-MiniLM-L6-v2`) so no extra API key; allow swapping to Voyage AI. Provide
  a TF-IDF fallback for a zero-ML-dependency build.
- **Retrieve:** embed `(request + subtask_desc)`, cosine vs stored vectors,
  return top-k. Feed those records' `fix_summary` + `final_code` to the planner
  (for decomposition) and the worker (as few-shot guidance).
- **Write:** after each subtask resolves, store what failed and what fixed it.
  This is the highest-value field — capture the *delta*, not just the result.

---

## Config (`config.py`)

```python
WORKER_MODEL     = "claude-sonnet-4-6"
JUDGE_MODEL      = "claude-opus-4-8"     # different from worker on purpose
PLANNER_MODEL    = "claude-sonnet-4-6"
ESCALATION_MODEL = "claude-opus-4-8"     # used when a subtask plateaus
MAX_ITERATIONS   = 4
SCORE_THRESHOLD  = 8
PLATEAU_PATIENCE = 2
EXEC_TIMEOUT     = 10
TOKEN_BUDGET     = 200_000               # per request, cumulative
MEMORY_DB_PATH   = "./builder_memory.db"
MEMORY_TOP_K     = 3
```
(Verify the exact current model strings against the Anthropic docs at build time.)

---

## Build milestones (build + test in this order)

- **M1 — Foundations.** `config`, `schemas`, `llm` wrapper (mockable), `sandbox`.
  Test: dataclasses round-trip; sandbox runs good code (pass), bad code (fail),
  infinite loop (timeout). No live API needed.
- **M2 — Single-task loop.** `clarify`, `generate` + `self_critique`, `verify`
  (test-first + cross-model judge), and an `orchestrate` that runs ONE subtask.
  Test: with the LLM mocked, a passing verdict exits the loop; a failing one
  feeds issues back; the loop respects MAX_ITERATIONS.
- **M3 — Memory.** `memory.py` store/retrieve with the pluggable embedder.
  Test: store N records, retrieve returns the most similar; embedder swap works.
- **M4 — Planner + integration.** `plan`, `integrate`, and a multi-subtask
  `orchestrate` with dependency ordering + final verify. Test: a 2-subtask plan
  builds, integrates, and final-verifies.
- **M5 — Guards + CLI.** Plateau detection, token budget, escalation, `cli.py`.
  Test: plateau triggers escalation; budget abort returns best-so-far.

Each milestone's acceptance bar: its unit tests pass and the loop behaves
correctly with the LLM mocked.

---

## Key trade-offs (make these explicit in code comments)

| Decision | Why | Revisit when |
|---|---|---|
| Raw SDK, no agent framework | Keeps control flow transparent | State/persistence grows → LangGraph |
| SQLite + local embeddings for memory | Zero infra, no extra API key | Scale/multi-user → a real vector DB |
| Cross-model judge | Independent blind spots | Judge unreliable → invest in tests |
| Subprocess sandbox | Fine for local dev | **Untrusted input → real sandbox** |
| Synchronous loop | Easy to reason about | Long builds → async/job queue |

**Security (non-negotiable):** the sandbox executes model-generated code via
subprocess + timeout — that is NOT isolation. Before this touches untrusted
input or runs anywhere sensitive, swap in a container with no network and
resource caps.

**Biggest lever:** the more of "is it good?" you convert into "does it pass this
check?", the more reliable the whole loop becomes. For data/SQL/pipeline output
types, make the verifier domain-specific — schema checks, row-count/null
assertions, dbt tests, `EXPLAIN` plans — instead of leaning on the LLM judge.

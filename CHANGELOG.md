# Changelog

## v0.2.0

### Added
- **Resumable builds**: multi-subtask builds checkpoint progress after every
  subtask. Pass `--resume` to `whetstone build "<same request>"` to pick up
  where a failed or interrupted build left off, without re-running subtasks
  that already passed.
- **Agent loop trace**: every generate → critique → verify iteration for a
  subtask is now kept (not just the best attempt). Inspect it in the REPL
  with `/trace <build#> <subtask_id>`, or read it straight off
  `subtask_results[id]["attempts"]` in `--json` output.
- **Retry with backoff**: transient provider failures (rate limits, dropped
  connections, 5xx) are now detected by exception type across the
  OpenAI/Anthropic/httpx SDKs and retried with exponential backoff instead
  of failing the build outright.
- **Cost tracking**: builds track estimated USD cost per model via
  `MODEL_PRICING`, configurable through `.whetstone.toml`'s `[pricing]`
  table; budgets can cap on cost as well as token count.

### Fixed
- `MAX_RETRIES` / `RETRY_DELAY` were exposed as TOML config keys but had no
  module-level defaults and no code path ever read them — now wired up end
  to end.
- Release/publish CI installed `.[dev]`, which skips the optional
  `anthropic` extra and broke `test_anthropic_*` in CI. Now installs
  `.[all]`.

## v0.1.0

Initial release.

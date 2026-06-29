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

### Fixed
- `MAX_RETRIES` / `RETRY_DELAY` were exposed as TOML config keys but had no
  module-level defaults and no code path actually used them — `llm.ask()`
  and `llm.embed()` now retry transient provider failures (connection drops,
  rate limits) with exponential backoff before giving up.

## v0.1.0

Initial release.

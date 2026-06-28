# Tutorials

This section provides tutorials for extending and customizing Whetstone.

## Interactive Clarification

By default, Whetstone runs in interactive clarification mode. When you request a build, the analyst agent will inspect your query and ask up to three clarifying questions to refine the specification details.

You can toggle interactive mode on and off in the REPL using:

```text
❯ /clarify off
```

When interactive clarification is disabled, Whetstone compiles requests using sensible default assumptions.

## Controlling the Budget

To prevent infinite loops or burning tokens on complex failures, you can configure cumulative token limits.

Specify a budget constraint when executing CLI runs:

```bash
whetstone --max-cost 0.50 "build a custom parser"
```

If the build cost exceeds the maximum threshold, Whetstone aborts the state machine iteration loop and exports the best attempt code produced up to that execution block.

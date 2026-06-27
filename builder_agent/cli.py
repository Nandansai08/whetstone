from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, is_dataclass

from builder_agent import config
from builder_agent.budget import TokenBudget
from builder_agent.llm import set_budget
from builder_agent.memory import Memory
from builder_agent.orchestrate import orchestrate

# ── Silence noisy HTTP loggers at import time ────────────────────────

for _n in ("httpx", "httpcore", "openai", "anthropic"):
    _lg = logging.getLogger(_n)
    _lg.setLevel(logging.WARNING)
    _lg.propagate = False


# ── Constants ────────────────────────────────────────────────────────

BANNER = r"""
 __        ___          _       _
 \ \      / / |__   ___| |_ ___| |_ ___  _ __   ___
  \ \ /\ / /| '_ \ / _ \ __/ __| __/ _ \| '_ \ / _ \
   \ V  V / | | | |  __/ |_\__ \ || (_) | | | |  __/
    \_/\_/  |_| |_|\___|\__|___/\__\___/|_| |_|\___|
""".strip()

VERSION = "0.1.0"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ABORTED = 2
EXIT_USAGE = 3

# ── ANSI helpers ─────────────────────────────────────────────────────

_NO_COLOR = os.environ.get("NO_COLOR") is not None
_TTY = sys.stdout.isatty()
_COLOR = not _NO_COLOR and _TTY


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str:
    return _c("2", t)


def bold(t: str) -> str:
    return _c("1", t)


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def magenta(t: str) -> str:
    return _c("35", t)


def blue(t: str) -> str:
    return _c("34", t)


def _clear_line() -> None:
    if _TTY:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


# ── Spinner ──────────────────────────────────────────────────────────


class Spinner:
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._message = ""
        self._detail = ""
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._start_time = 0.0

    def _animate(self) -> None:
        frames = itertools.cycle(self._FRAMES)
        while self._running:
            with self._lock:
                msg = self._message
                detail = self._detail
            elapsed = time.time() - self._start_time
            timer = dim(f" {elapsed:.0f}s")
            det = f" {dim(detail)}" if detail else ""
            if _TTY:
                frame = next(frames)
                sys.stdout.write(
                    f"\r  {cyan(frame)} {msg}{det}{timer}\033[K"
                )
                sys.stdout.flush()
            time.sleep(0.08)

    def start(self, message: str, detail: str = "") -> None:
        self.stop()
        self._message = message
        self._detail = detail
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def update(self, message: str, detail: str = "") -> None:
        with self._lock:
            self._message = message
            if detail:
                self._detail = detail

    def stop(self, final: str = "") -> None:
        was_running = self._running
        self._running = False
        if self._thread:
            self._thread.join()
            self._thread = None
        if was_running or final:
            _clear_line()
        if final:
            print(f"  {final}")

    def elapsed(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0


# ── Progress renderer (callback-driven) ─────────────────────────────


class ProgressRenderer:
    def __init__(self, spinner: Spinner):
        self._spinner = spinner
        self._stage_start = 0.0
        self._start_of_line = True

    def _elapsed_tag(self) -> str:
        dt = time.time() - self._stage_start
        return dim(f"({dt:.1f}s)")

    def __call__(self, event: str, data: dict) -> None:
        s = self._spinner

        if event == "clarifying":
            self._stage_start = time.time()
            s.start("Clarifying request")

        elif event == "clarified":
            desc = data["description"]
            if len(desc) > 70:
                desc = desc[:67] + "..."
            s.stop(f"{green('✓')} Clarified {self._elapsed_tag()}")
            print(f"    {dim(desc)}")

        elif event == "planning":
            self._stage_start = time.time()
            s.start("Planning subtasks")

        elif event == "planned":
            count = data["count"]
            s.stop(
                f"{green('✓')} Plan: {bold(str(count))} "
                f"subtask{'s' if count != 1 else ''} "
                f"{self._elapsed_tag()}"
            )
            for st in data["subtasks"]:
                print(f"    {cyan(st['id'])} {dim(st['description'])}")

        elif event == "subtask_start":
            idx = data["index"]
            total = data["total"]
            sid = data["subtask"]
            desc = data["description"]
            self._stage_start = time.time()
            tag = f"[{idx + 1}/{total}]"
            print()
            print(f"  {bold(tag)} {cyan(sid)} {desc}")

        elif event == "generating":
            it = data["iteration"]
            s.stop()
            print(f"  {bold('Generating')} {dim(f'iter {it}')}:")
            self._start_of_line = True

        elif event == "chunk":
            chunk = data["chunk"]
            parts = chunk.split("\n")
            for i, part in enumerate(parts):
                if i > 0:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    self._start_of_line = True
                if part:
                    if self._start_of_line:
                        sys.stdout.write("    ")
                        self._start_of_line = False
                    sys.stdout.write(part)
            sys.stdout.flush()

        elif event == "critiquing":
            print()
            s.start("Self-critiquing")

        elif event == "verifying":
            s.update("Verifying")

        elif event == "verdict":
            score = data["score"]
            passed = data["passed"]
            it = data["iteration"]
            dt = self._elapsed_tag()
            if passed:
                s.stop(
                    f"{green('✓')} iter {it} "
                    f"score {green(bold(f'{score}/10'))} {dt}"
                )
            else:
                issues = data.get("issues", [])
                s.stop(
                    f"{yellow('○')} iter {it} "
                    f"score {yellow(f'{score}/10')} {dt}"
                )
                for issue in issues[:2]:
                    if len(issue) > 72:
                        issue = issue[:69] + "..."
                    print(f"      {dim('·')} {dim(issue)}")

        elif event == "escalating":
            model = data.get("model", "?")
            s.stop(f"{yellow('⚡')} Escalating → {yellow(model)}")

        elif event == "plateau_stuck":
            s.stop(f"{red('✗')} Stuck after escalation")

        elif event == "budget_exceeded":
            s.stop(f"{yellow('⚠')}  Token budget exceeded")

        elif event == "subtask_done":
            pass

        elif event == "integrating":
            self._stage_start = time.time()
            s.start("Integrating code")

        elif event == "final_verify":
            s.update("Final verification")


# ── Display helpers ──────────────────────────────────────────────────


def _to_jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _print_banner():
    print()
    print(cyan(BANNER))
    tag = dim(f"v{VERSION}")
    model = dim(config.WORKER_MODEL.model_id)
    print(f"  {bold('Builder Agent')} {tag}  {dim('·')}  {model}")
    print()


def _code_block(code: str, *, numbered: bool = True) -> None:
    lines = code.strip().splitlines()
    width = len(str(len(lines)))
    print(f"  {dim('┌' + '─' * 60)}")
    for i, line in enumerate(lines, 1):
        if numbered:
            num = dim(f"{i:>{width}} │ ")
        else:
            num = dim("│ ")
        print(f"  {num}{line}")
    print(f"  {dim('└' + '─' * 60)}")


def _score_bar(score: int, threshold: int) -> str:
    bar_len = 20
    filled = int(score / 10 * bar_len)
    if score >= threshold:
        color = green
    elif score >= threshold - 2:
        color = yellow
    else:
        color = red
    bar = color("█" * filled) + dim("░" * (bar_len - filled))
    return f"[{bar}] {bold(str(score))}/10"


def _print_result(result: dict, output_path: str = "") -> None:
    fv = result.get("final_verdict")
    succeeded = result["succeeded"]

    print()
    sep = dim("─" * 50)
    print(f"  {sep}")

    if succeeded:
        print(f"  {green('●')} {bold(green('BUILD PASSED'))}")
    elif result.get("aborted_reason"):
        reason = result["aborted_reason"]
        print(f"  {yellow('●')} {bold(yellow(f'ABORTED: {reason}'))}")
    else:
        print(f"  {red('●')} {bold(red('BUILD FAILED'))}")

    if fv:
        bar = _score_bar(fv.score, config.SCORE_THRESHOLD)
        print(f"    Score  {bar}")
        if fv.issues:
            for issue in fv.issues:
                print(f"      {dim('·')} {issue}")

    sr = result.get("subtask_results", {})
    if sr:
        iters = sum(r.get("iterations", 0) for r in sr.values())
        esc = sum(1 for r in sr.values() if r.get("escalated"))
        parts = [f"{len(sr)} subtasks", f"{iters} iters"]
        if esc:
            parts.append(f"{esc} escalated")
        print(f"    Stats  {dim(' · '.join(parts))}")

    if result.get("usage"):
        u = result["usage"]
        tok = f"{u['total_tokens']:,}"
        lim = f"{u['limit']:,}"
        print(f"    Tokens {dim(f'{tok} / {lim}')}")

    artifact = result.get("artifact")
    if artifact:
        lines = artifact.strip().splitlines()
        print(f"\n  {bold('Output')} {dim(f'{len(lines)} lines')}")
        _code_block(artifact)

    if output_path and artifact:
        with open(output_path, "w") as f:
            f.write(artifact)
        print(f"\n  {green('→')} Saved to {bold(output_path)}")


def _print_help():
    cmds = [
        (cyan("<request>"), "Build something"),
        (cyan("/config"), "Show model configuration"),
        (cyan("/memory"), "List stored memory records"),
        (cyan("/memory show <id>"), "Show a specific record"),
        (cyan("/memory clear"), "Clear all records"),
        (cyan("/help"), "Show this help"),
        (cyan("/quit"), "Exit"),
    ]
    print()
    print(f"  {bold('Commands')}")
    for cmd, desc in cmds:
        print(f"    {cmd:<30s} {dim(desc)}")
    print()
    print(f"  {bold('Examples')}")
    print(dim("    Build a function add(a, b) that returns a + b."))
    print(dim("    Build a CSV parser with custom delimiters."))
    print(dim("    Build a binary search that returns the index."))
    print()


def _print_config():
    models = [
        ("worker", config.WORKER_MODEL),
        ("judge", config.JUDGE_MODEL),
        ("planner", config.PLANNER_MODEL),
        ("escalation", config.ESCALATION_MODEL),
    ]
    settings = [
        ("threshold", f"{config.SCORE_THRESHOLD}/10"),
        ("max_iter", str(config.MAX_ITERATIONS)),
        ("patience", str(config.PLATEAU_PATIENCE)),
        ("budget", f"{config.TOKEN_BUDGET:,} tokens"),
        ("memory", config.MEMORY_DB_PATH),
    ]

    print()
    print(f"  {bold('Models')}")
    for label, m in models:
        prov = dim(f"({m.provider})")
        print(f"    {label:<12s} {cyan(m.model_id)} {prov}")
    if config.WORKER_MODEL.base_url:
        print(f"    {'endpoint':<12s} {dim(config.WORKER_MODEL.base_url)}")

    print()
    print(f"  {bold('Settings')}")
    for label, val in settings:
        print(f"    {label:<12s} {val}")
    print()


def _handle_memory_command(parts: list[str], memory: Memory) -> None:
    if len(parts) == 1 or parts[1] == "list":
        rtype = None
        if len(parts) > 2 and parts[2] in ("subtask", "plan"):
            rtype = parts[2]
        records = memory.list_records(record_type=rtype)
        if not records:
            print(dim("  No records."))
            return
        print()
        for r in records:
            icon = blue("◆") if r["record_type"] == "plan" else dim("◇")
            rid = bold(str(r["id"]).rjust(3))
            req = r["request"][:42].ljust(42)
            ts = dim(r["created_at"][:16])
            print(f"  {icon} {rid} {req} {ts}")
        print()
    elif parts[1] == "show" and len(parts) > 2:
        try:
            record_id = int(parts[2])
        except ValueError:
            print(red("  Invalid ID."))
            return
        rec = memory.get_record(record_id)
        if rec is None:
            print(red(f"  Record {record_id} not found."))
            return
        print(f"\n  {bold('Record')} {cyan(str(rec['id']))}")
        print(f"    request   {rec['request']}")
        print(f"    type      {rec['record_type']}")
        print(f"    subtask   {rec['subtask_desc']}")
        print(f"    fix       {rec['fix_summary']}")
        if rec["failures"]:
            print("    failures")
            for f in rec["failures"]:
                print(f"      {dim('·')} {f}")
        if rec["final_code"]:
            print()
            _code_block(rec["final_code"])
        print()
    elif parts[1] == "clear":
        answer = input(
            f"  {yellow('Delete all records?')} [y/N] "
        ).strip().lower()
        if answer == "y":
            count = memory.clear()
            print(f"  {green('✓')} Deleted {count} records.")
        else:
            print(dim("  Cancelled."))
    else:
        print(dim("  Usage: /memory [list|show <id>|clear]"))


# ── Interactive REPL ─────────────────────────────────────────────────


def _setup_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    orch_logger = logging.getLogger("builder_agent.orchestrate")
    orch_logger.handlers.clear()
    orch_logger.setLevel(logging.WARNING)
    orch_logger.propagate = False


def _repl() -> int:
    _setup_logging()
    _print_banner()

    memory = Memory()
    build_count = 0

    print(dim("  Type what you want to build, or /help for commands."))
    print()

    while True:
        try:
            prompt = input(f"  {bold(cyan('❯'))} ")
        except (EOFError, KeyboardInterrupt):
            print()
            print(dim("  Goodbye."))
            return EXIT_SUCCESS

        prompt = prompt.strip()
        if not prompt:
            continue

        if prompt in ("/quit", "/exit", "/q"):
            print(dim("  Goodbye."))
            return EXIT_SUCCESS

        if prompt == "/help":
            _print_help()
            continue

        if prompt == "/config":
            _print_config()
            continue

        if prompt.startswith("/memory"):
            parts = prompt.split()
            _handle_memory_command(parts, memory)
            continue

        if prompt.startswith("/"):
            print(f"  {red('?')} Unknown command: {prompt.split()[0]}")
            print(dim("    Type /help for available commands."))
            continue

        build_count += 1
        print()
        print(f"  {dim('─' * 50)}")
        print(f"  {bold(f'Build #{build_count}')}")
        print(f"  {dim('─' * 50)}")

        spinner = Spinner()
        progress = ProgressRenderer(spinner)

        budget = TokenBudget(limit=config.TOKEN_BUDGET)
        set_budget(budget)

        start = time.time()
        try:
            result = orchestrate(
                prompt,
                interactive=False,
                memory=memory,
                budget=budget,
                on_progress=progress,
            )
        except KeyboardInterrupt:
            spinner.stop(f"{yellow('⚠')}  Interrupted")
            print()
            continue
        except Exception as e:
            spinner.stop(f"{red('✗')} Error: {e}")
            print()
            continue
        elapsed = time.time() - start

        spinner.stop()
        _print_result(result)
        print(f"    Time   {dim(f'{elapsed:.1f}s')}")
        print()


# ── One-shot subcommands ─────────────────────────────────────────────


def _cmd_build(args) -> int:
    if not args.json:
        _print_banner()

    _setup_logging()

    spinner = Spinner()
    progress = ProgressRenderer(spinner) if not args.json else None

    if args.max_iterations:
        config.MAX_ITERATIONS = args.max_iterations

    budget = None
    token_limit = args.token_budget or config.TOKEN_BUDGET
    if token_limit > 0:
        budget = TokenBudget(limit=token_limit)
        set_budget(budget)

    memory = None
    if not args.no_memory:
        memory = Memory()

    if not args.json:
        spinner.start("Starting build")

    result = orchestrate(
        args.request,
        interactive=not args.non_interactive,
        memory=memory,
        budget=budget,
        on_progress=progress or (lambda e, d: None),
    )

    if not args.json:
        spinner.stop()

    if args.json:
        print(json.dumps(_to_jsonable(result), indent=2, default=str))
    else:
        _print_result(result, output_path=args.output)

    if result.get("aborted_reason"):
        return EXIT_ABORTED
    return EXIT_SUCCESS if result["succeeded"] else EXIT_FAILURE


def _cmd_memory_list(args) -> int:
    mem = Memory()
    records = mem.list_records(record_type=args.type)
    if not records:
        print("No records.")
        return EXIT_SUCCESS
    for r in records:
        print(
            f"  [{r['id']}] {r['record_type']:8s} "
            f"{r['request'][:40]:40s} {r['created_at']}"
        )
    return EXIT_SUCCESS


def _cmd_memory_show(args) -> int:
    mem = Memory()
    record = mem.get_record(args.id)
    if record is None:
        print(f"Record {args.id} not found.")
        return EXIT_FAILURE
    print(json.dumps(record, indent=2))
    return EXIT_SUCCESS


def _cmd_memory_clear(args) -> int:
    if not args.yes:
        answer = input("Delete all memory records? [y/N] ").strip().lower()
        if answer != "y":
            print("Cancelled.")
            return EXIT_SUCCESS
    mem = Memory()
    count = mem.clear()
    print(f"Deleted {count} records.")
    return EXIT_SUCCESS


# ── Entrypoint ───────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        prog="whetstone",
        description="Whetstone — AI-powered code builder",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="Interactive REPL (default)")

    build_p = sub.add_parser("build", help="One-shot build")
    build_p.add_argument("request", help="What to build")
    build_p.add_argument(
        "--non-interactive", action="store_true",
        help="Skip clarifying questions",
    )
    build_p.add_argument(
        "--max-iterations", type=int, default=0,
        help="Override max iterations per subtask",
    )
    build_p.add_argument(
        "--token-budget", type=int, default=0,
        help="Token budget (0 = use config default)",
    )
    build_p.add_argument(
        "--no-memory", action="store_true",
        help="Skip memory read/write",
    )
    build_p.add_argument(
        "--output", type=str, default="",
        help="Write artifact to file",
    )
    build_p.add_argument(
        "--json", action="store_true",
        help="Emit structured JSON result",
    )

    mem_p = sub.add_parser("memory", help="Manage memory records")
    mem_sub = mem_p.add_subparsers(dest="mem_command")

    list_p = mem_sub.add_parser("list", help="List records")
    list_p.add_argument(
        "--type", choices=["subtask", "plan"], default=None,
    )

    show_p = mem_sub.add_parser("show", help="Show a record")
    show_p.add_argument("id", type=int, help="Record ID")

    clear_p = mem_sub.add_parser("clear", help="Clear all records")
    clear_p.add_argument(
        "--yes", action="store_true", help="Skip confirmation",
    )

    args = parser.parse_args(argv)

    if args.command is None or args.command == "chat":
        return _repl()
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "memory":
        if args.mem_command == "list":
            return _cmd_memory_list(args)
        if args.mem_command == "show":
            return _cmd_memory_show(args)
        if args.mem_command == "clear":
            return _cmd_memory_clear(args)
        mem_p.print_help()
        return EXIT_USAGE

    parser.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())

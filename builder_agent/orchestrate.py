from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

from builder_agent import config
from builder_agent.budget import TokenBudget
from builder_agent.clarify import clarify
from builder_agent.generate import generate, self_critique
from builder_agent.integrate import integrate
from builder_agent.llm import ask
from builder_agent.memory import Memory
from builder_agent.plan import plan as make_plan
from builder_agent.schemas import (
    Attempt,
    MemoryRecord,
    Plan,
    Spec,
    SubTask,
)
from builder_agent.verify import verify

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, dict], None]

_db_lock = threading.Lock()


def _noop_progress(event: str, data: dict) -> None:
    pass


_FIX_SUMMARY_SYSTEM = (
    "Summarize what changed between the failing and passing code, "
    "and why it fixed the failure. Max 200 chars. No markdown."
)

_FIX_SUMMARY_PROMPT = (
    "Failing code:\n{failing_code}\n\n"
    "Failure issues:\n{issues}\n\n"
    "Passing code:\n{passing_code}\n\n"
    "Summarize the fix in under 200 characters."
)


def _make_fix_summary(
    failing_code: str, failing_issues: list[str], passing_code: str
) -> str:
    prompt = _FIX_SUMMARY_PROMPT.format(
        failing_code=failing_code,
        issues="\n".join(failing_issues),
        passing_code=passing_code,
    )
    return ask(prompt, model=config.WORKER_MODEL, system=_FIX_SUMMARY_SYSTEM)


def _store_subtask_memory(
    memory: Memory,
    spec: Spec,
    subtask: SubTask,
    attempts: list[Attempt],
    best: Attempt,
) -> None:
    all_failures = []
    first_failing_code = ""
    for a in attempts:
        if not a.verdict.passed:
            all_failures.extend(a.verdict.issues)
            if not first_failing_code:
                first_failing_code = a.code

    if first_failing_code and best.code != first_failing_code:
        fix_summary = _make_fix_summary(
            first_failing_code, all_failures, best.code
        )
    else:
        fix_summary = "Passed on first attempt"

    embedding = memory._embedder.embed(
        spec.request + " " + subtask.description
    )
    record = MemoryRecord(
        request=spec.request,
        output_type=spec.output_type,
        subtask_desc=subtask.description,
        failures=all_failures,
        fix_summary=fix_summary,
        final_code=best.code,
        embedding=embedding,
        record_type="subtask",
    )
    with _db_lock:
        memory.store(record)


def _store_plan_memory(
    memory: Memory,
    spec: Spec,
    plan_desc: str,
    final_passed: bool,
) -> None:
    embedding = memory._embedder.embed(spec.request)
    outcome = "final verify passed" if final_passed else "final verify failed"
    record = MemoryRecord(
        request=spec.request,
        output_type=spec.output_type,
        subtask_desc=plan_desc,
        failures=[] if final_passed else ["final integration verify failed"],
        fix_summary=outcome,
        final_code="",
        embedding=embedding,
        record_type="plan",
    )
    with _db_lock:
        memory.store(record)


def _detect_plateau(scores: list[int], patience: int) -> bool:
    if len(scores) < patience + 1:
        return False
    window = scores[-patience:]
    before = scores[:-patience]
    return max(window) <= max(before)


def orchestrate_subtask(
    subtask: SubTask,
    spec: Spec,
    memory: Memory | None = None,
    budget: TokenBudget | None = None,
    on_progress: ProgressCallback = _noop_progress,
) -> dict:
    best: Attempt | None = None
    feedback: str | None = None
    attempts: list[Attempt] = []
    scores: list[int] = []
    escalated = False
    aborted_reason: str | None = None
    current_worker = config.WORKER_MODEL

    memory_hints = None
    if memory is not None:
        memory_hints = memory.retrieve(
            subtask.description, k=config.MEMORY_TOP_K,
            record_type="subtask",
        )
        if not memory_hints:
            memory_hints = None

    for i in range(config.MAX_ITERATIONS):
        if budget and budget.exceeded():
            aborted_reason = "token_budget"
            on_progress("budget_exceeded", {"subtask": subtask.id})
            logger.info("[%s] token budget exceeded, aborting", subtask.id)
            break

        if (
            not escalated
            and _detect_plateau(scores, config.PLATEAU_PATIENCE)
        ):
            escalated = True
            current_worker = config.ESCALATION_MODEL
            on_progress("escalating", {
                "subtask": subtask.id,
                "model": current_worker.model_id,
            })
            logger.info(
                "[%s] plateau detected at iter %d, escalating to %s",
                subtask.id, i + 1, current_worker.model_id,
            )

        if (
            escalated
            and len(scores) > config.PLATEAU_PATIENCE + 1
            and _detect_plateau(scores, 1)
        ):
            aborted_reason = "plateau"
            on_progress("plateau_stuck", {"subtask": subtask.id})
            logger.info(
                "[%s] still stuck after escalation, stopping", subtask.id,
            )
            break

        on_progress("generating", {
            "subtask": subtask.id, "iteration": i + 1,
        })
        logger.info("[%s] iter %d — generating code", subtask.id, i + 1)

        def chunk_callback(chunk: str) -> None:
            on_progress("chunk", {
                "subtask": subtask.id,
                "iteration": i + 1,
                "chunk": chunk,
            })

        code = generate(
            subtask, spec, feedback=feedback,
            memory_hints=memory_hints,
            worker_model=current_worker,
            on_chunk=chunk_callback,
        )

        on_progress("critiquing", {
            "subtask": subtask.id, "iteration": i + 1,
        })
        logger.info("[%s] iter %d — self-critique", subtask.id, i + 1)
        code = self_critique(code, subtask, worker_model=current_worker)

        on_progress("verifying", {
            "subtask": subtask.id, "iteration": i + 1,
        })
        logger.info("[%s] iter %d — verifying", subtask.id, i + 1)
        verdict = verify(subtask, code, output_type=spec.output_type)

        on_progress("verdict", {
            "subtask": subtask.id,
            "iteration": i + 1,
            "score": verdict.score,
            "passed": verdict.passed,
            "issues": verdict.issues,
        })
        logger.info(
            "[%s] iter %d — score=%d passed=%s",
            subtask.id, i + 1, verdict.score, verdict.passed,
        )
        attempt = Attempt(iteration=i, code=code, verdict=verdict)
        attempts.append(attempt)
        scores.append(verdict.score)

        if best is None or verdict.score > best.verdict.score:
            best = attempt

        if verdict.passed:
            if memory is not None:
                _store_subtask_memory(memory, spec, subtask, attempts, best)
            return {
                "succeeded": True,
                "attempt": best,
                "iterations": i + 1,
                "escalated": escalated,
                "aborted_reason": None,
            }

        feedback = "\n".join(verdict.issues)

    if memory is not None and best is not None:
        _store_subtask_memory(memory, spec, subtask, attempts, best)

    return {
        "succeeded": False,
        "attempt": best,
        "iterations": len(attempts),
        "escalated": escalated,
        "aborted_reason": aborted_reason,
    }


def _run_async(coro):
    try:
        # asyncio.get_running_loop() raises RuntimeError specifically if there is
        # no active event loop running in the current thread. This is safe to catch
        # here as it serves as the standard loop detection check.
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


async def _async_orchestrate(
    spec: Spec,
    the_plan: Plan,
    memory: Memory | None,
    budget: TokenBudget | None,
    on_progress: ProgressCallback,
) -> dict:
    all_plan_ids = {st.id for st in the_plan.subtasks}
    in_degree = {
        st.id: sum(1 for d in st.depends_on if d in all_plan_ids)
        for st in the_plan.subtasks
    }
    adjacency = {st.id: [] for st in the_plan.subtasks}
    for st in the_plan.subtasks:
        for d in st.depends_on:
            if d in all_plan_ids:
                adjacency[d].append(st.id)

    subtask_by_id = {st.id: st for st in the_plan.subtasks}
    total = len(the_plan.subtasks)
    subtask_indices = {st.id: i for i, st in enumerate(the_plan.subtasks)}

    outputs: dict[str, str] = {}
    subtask_results: dict[str, dict] = {}

    ready_ids = [sid for sid, deg in in_degree.items() if deg == 0]
    ready_ids.sort(key=lambda x: subtask_indices[x])

    active_tasks = {}  # Task -> str (subtask id)
    first_failure_id = None
    first_failure_reason = None
    any_failed = False

    # Note on Cancellation Tradeoff:
    # Python threads running inside asyncio.to_thread (orchestrate_subtask) do not
    # support forceful cancellation/termination. Attempting to cancel the asyncio
    # task wrapper would leave the underlying synchronous executor thread running
    # in the background (potentially performing unsafe/concurrent LLM and sandbox
    # operations). Thus, we do not forcefully cancel active tasks upon failure.
    # Instead, we prevent scheduling any new subtasks (by checking not
    # any_failed before scheduling) and wait for already in-flight subtasks
    # to complete gracefully.
    while ready_ids or active_tasks:
        while ready_ids and not any_failed:
            st_id = ready_ids.pop(0)
            subtask = subtask_by_id[st_id]

            if budget and budget.exceeded():
                if not any_failed:
                    any_failed = True
                    first_failure_id = st_id
                    first_failure_reason = "token_budget"
                subtask_results[st_id] = {
                    "succeeded": False,
                    "attempt": None,
                    "iterations": 0,
                    "escalated": False,
                    "aborted_reason": "token_budget",
                }
                on_progress("budget_exceeded", {"subtask": st_id})
                continue

            idx = subtask_indices[st_id]
            on_progress("subtask_start", {
                "subtask": st_id,
                "description": subtask.description,
                "index": idx,
                "total": total,
            })

            task = asyncio.create_task(
                asyncio.to_thread(
                    orchestrate_subtask,
                    subtask,
                    spec,
                    memory=memory,
                    budget=budget,
                    on_progress=on_progress,
                )
            )
            active_tasks[task] = st_id

        if not active_tasks:
            break

        done, _ = await asyncio.wait(
            active_tasks.keys(),
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            st_id = active_tasks.pop(task)
            idx = subtask_indices[st_id]

            try:
                result = task.result()
            except Exception as e:
                logger.error("Subtask %s raised exception: %s", st_id, e)
                result = {
                    "succeeded": False,
                    "attempt": None,
                    "iterations": 0,
                    "escalated": False,
                    "aborted_reason": str(e),
                }

            subtask_results[st_id] = result

            on_progress("subtask_done", {
                "subtask": st_id,
                "succeeded": result["succeeded"],
                "iterations": result["iterations"],
                "index": idx,
                "total": total,
            })

            if result["succeeded"]:
                outputs[st_id] = result["attempt"].code
                deps = sorted(adjacency[st_id], key=lambda x: subtask_indices[x])
                for dep_id in deps:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        ready_ids.append(dep_id)
            else:
                if not any_failed:
                    any_failed = True
                    first_failure_id = st_id
                    first_failure_reason = result.get("aborted_reason")

    if len(subtask_results) < total and not any_failed:
        for st in the_plan.subtasks:
            if st.id not in subtask_results:
                any_failed = True
                first_failure_id = st.id
                first_failure_reason = "dependency_failed"
                break

    return {
        "succeeded": not any_failed,
        "first_failure_id": first_failure_id,
        "first_failure_reason": first_failure_reason,
        "outputs": outputs,
        "subtask_results": subtask_results,
    }


def orchestrate(
    request: str,
    *,
    memory: Memory | None = None,
    interactive: bool = True,
    budget: TokenBudget | None = None,
    on_progress: ProgressCallback = _noop_progress,
) -> dict:
    on_progress("clarifying", {})
    logger.info("Clarifying request...")
    spec = clarify(request, interactive=interactive)
    on_progress("clarified", {"description": spec.description})
    logger.info("Spec: %s", spec.description)

    on_progress("planning", {})
    logger.info("Planning...")
    the_plan = make_plan(spec, memory=memory)
    on_progress("planned", {
        "count": len(the_plan.subtasks),
        "ids": [s.id for s in the_plan.subtasks],
        "subtasks": [
            {"id": s.id, "description": s.description}
            for s in the_plan.subtasks
        ],
    })
    logger.info(
        "Plan: %d subtasks — %s",
        len(the_plan.subtasks),
        ", ".join(s.id for s in the_plan.subtasks),
    )

    res = _run_async(
        _async_orchestrate(
            spec,
            the_plan,
            memory=memory,
            budget=budget,
            on_progress=on_progress,
        )
    )

    subtask_results = res["subtask_results"]

    if not res["succeeded"]:
        if memory is not None:
            plan_desc = " -> ".join(s.id for s in the_plan.subtasks)
            _store_plan_memory(memory, spec, plan_desc, False)
        return {
            "succeeded": False,
            "halted_at": res["first_failure_id"],
            "plan": the_plan,
            "spec": spec,
            "subtask_results": subtask_results,
            "artifact": None,
            "final_verdict": None,
            "aborted_reason": res["first_failure_reason"],
            "usage": budget.usage() if budget else None,
        }

    on_progress("integrating", {})
    logger.info("Integrating outputs...")
    artifact = integrate(spec, res["outputs"], the_plan)

    on_progress("final_verify", {})
    logger.info("Running final verification...")

    final_subtask = SubTask(
        id="_final_verify",
        description="Final integration verification",
        acceptance_criteria=spec.acceptance_criteria,
    )
    final_verdict = verify(final_subtask, artifact, output_type=spec.output_type)

    if memory is not None:
        plan_desc = " -> ".join(s.id for s in the_plan.subtasks)
        _store_plan_memory(memory, spec, plan_desc, final_verdict.passed)

    return {
        "succeeded": final_verdict.passed,
        "halted_at": None,
        "plan": the_plan,
        "spec": spec,
        "subtask_results": subtask_results,
        "artifact": artifact,
        "final_verdict": final_verdict,
        "aborted_reason": None,
        "usage": budget.usage() if budget else None,
    }

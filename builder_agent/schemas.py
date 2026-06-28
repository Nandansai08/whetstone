"""Core dataclasses and structural schemas representing data contracts."""

from dataclasses import dataclass, field


@dataclass
class Spec:
    """System specifications representing a clarified user build request.

    Attributes:
        request: Raw user prompt instruction string.
        description: Structured target description prose.
        acceptance_criteria: objective conditions required to verify the build.
        assumptions: Discovered constraints and default setup decisions.
        output_type: Kind of output artifact ("python_module", "sql", etc.).
    """
    request: str
    description: str
    acceptance_criteria: list[str]
    assumptions: list[str]
    output_type: str


@dataclass
class SubTask:
    """An individual build unit of work inside a decomposition plan.

    Attributes:
        id: Unique identifier for the subtask.
        description: Detailed task description.
        acceptance_criteria: Criteria required for this unit's pass verdict.
        depends_on: Prerequisites subtasks list.
    """
    id: str
    description: str
    acceptance_criteria: list[str]
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """Topologically sorted plan outlining execution sequences of subtasks.

    Attributes:
        subtasks: Ordered list of subtasks.
    """
    subtasks: list[SubTask]


@dataclass
class Verdict:
    """Result status produced by the verification and evaluation suite.

    Attributes:
        passed: True if both tests passed and the judge score meets threshold.
        score: Evaluated quality score (0 to 10).
        tests_passed: True if the executable tests passed successfully.
        issues: Identified code bugs, failures, or missed criteria feedback.
        exec_output: Standard stdout/stderr capture from verifier executions.
    """
    passed: bool
    score: int
    tests_passed: bool
    issues: list[str]
    exec_output: str


@dataclass
class Attempt:
    """A single code iteration cycle result.

    Attributes:
        iteration: Index of current iteration.
        code: Generated script content.
        verdict: Verification verdict outcome metadata.
    """
    iteration: int
    code: str
    verdict: Verdict


@dataclass
class MemoryRecord:
    """Stored history record of resolved builds and subtask updates.

    Attributes:
        request: Target user prompt context.
        output_type: Output type strategy matching target language.
        subtask_desc: Target description of subtask block.
        failures: Failure history encountered before resolution.
        fix_summary: Key lessons lessons and steps taken to resolve bugs.
        final_code: Source code block produced on success.
        embedding: Dense float embedding vector representing the task.
        record_type: Kind of record stored ("subtask" or "plan").
    """
    request: str
    output_type: str
    subtask_desc: str
    failures: list[str]
    fix_summary: str
    final_code: str
    embedding: list[float]
    record_type: str = "subtask"  # "subtask" | "plan"

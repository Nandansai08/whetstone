from dataclasses import dataclass, field


@dataclass
class Spec:
    request: str
    description: str
    acceptance_criteria: list[str]
    assumptions: list[str]
    output_type: str


@dataclass
class SubTask:
    id: str
    description: str
    acceptance_criteria: list[str]
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    subtasks: list[SubTask]


@dataclass
class Verdict:
    passed: bool
    score: int
    tests_passed: bool
    issues: list[str]
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
    failures: list[str]
    fix_summary: str
    final_code: str
    embedding: list[float]
    record_type: str = "subtask"  # "subtask" | "plan"

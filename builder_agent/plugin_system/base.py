from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class PluginContext:
    """Shared execution context passed to all plugins."""
    workspace_dir: str
    output_type: str
    build_id: str | None = None


@dataclass
class GenerationContext:
    """Context holding subtask, spec, and memory details for custom generators."""
    subtask_id: str
    subtask_description: str
    acceptance_criteria: list[str]
    depends_on: list[str] = field(default_factory=list)
    spec_request: str = ""
    spec_description: str = ""
    feedback: str | None = None
    memory_hints: list[Any] | None = field(default_factory=list)


@dataclass
class PluginVerificationResult:
    """Result returned by custom verifier plugins, merged into built-in verdicts."""
    passed: bool
    issues: list[str]
    exec_output: str = ""
    blocking: bool = True


class VerifierPlugin(Protocol):
    """Protocol for verification plugins (runs extra verification steps)."""

    def verify(
        self, subtask: Any, code: str | dict[str, str], context: PluginContext
    ) -> PluginVerificationResult | None:
        """Run additional custom verification checks."""
        ...


class GeneratorPlugin(Protocol):
    """Protocol for generation plugins (custom code/content generators)."""

    def generate(
        self,
        gen_context: GenerationContext,
        context: PluginContext,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str | None:
        """Generate code; return code string if handled, otherwise None."""
        ...


class PostProcessorPlugin(Protocol):
    """Protocol for post-processor plugins (modify code/artifacts at hook points)."""

    def post_process_subtask(
        self, subtask: Any, code: str, context: PluginContext
    ) -> str:
        """Post-process subtask generated code before verification."""
        ...

    def post_process_artifact(
        self, spec: Any, code: str | dict[str, str], context: PluginContext
    ) -> str | dict[str, str]:
        """Post-process final integrated code artifact before integration testing."""
        ...

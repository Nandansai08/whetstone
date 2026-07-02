import logging
import threading
from typing import Any, Callable, List, Optional

from builder_agent import config
from builder_agent.plugin_system.base import (
    GenerationContext,
    PluginContext,
    PluginVerificationResult,
)
from builder_agent.plugin_system.discovery import (
    load_entry_point_plugins,
    load_local_plugins,
)

logger = logging.getLogger(__name__)


class PluginManager:
    """Central manager orchestrating plugin discovery, activation, and execution."""

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.generators = []
        self.verifiers = []
        self.post_processors = []
        self._discovered = False
        self._lock = threading.Lock()

    def discover_and_register(self) -> None:
        """Discover and load external/built-in plugins, filtering out disabled ones."""
        with self._lock:
            if self._discovered:
                return

            disabled_set = set(getattr(config, "PLUGINS_DISABLED", []))

            # Register built-ins first
            try:
                from builder_agent.plugin_system.builtins import LinterPlugin
                self._register_plugin_class(LinterPlugin, disabled_set)
            except Exception as e:
                logger.error("Failed to load built-in LinterPlugin: %s", e)

            # Register decorated plugins
            try:
                from builder_agent.plugin_system import registered_plugins
                for cls in registered_plugins:
                    self._register_plugin_class(cls, disabled_set)
            except Exception as e:
                logger.error("Failed to load decorated plugins: %s", e)

            # Discover entry points
            for cls in load_entry_point_plugins():
                self._register_plugin_class(cls, disabled_set)

            # Discover local directory plugins
            local_dir = getattr(config, "PLUGIN_DIR", self.plugin_dir)
            for cls in load_local_plugins(local_dir):
                self._register_plugin_class(cls, disabled_set)

            self._discovered = True

    def _register_plugin_class(self, cls: type, disabled_set: set[str]) -> None:
        name = cls.__name__
        if name in disabled_set:
            logger.info("Skipping disabled plugin: %s", name)
            return

        try:
            instance = cls()
            registered = False
            if hasattr(instance, "generate"):
                self.generators.append(instance)
                registered = True
            if hasattr(instance, "verify"):
                self.verifiers.append(instance)
                registered = True
            if hasattr(instance, "post_process_subtask") or hasattr(
                instance, "post_process_artifact"
            ):
                self.post_processors.append(instance)
                registered = True

            if registered:
                logger.info("Registered plugin: %s", name)
        except Exception as e:
            logger.error("Failed to instantiate plugin class %s: %s", name, e)

    def run_generators(
        self,
        gen_context: GenerationContext,
        context: PluginContext,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Run all active generator plugins.

        Returns the first non-None code generated, or None.
        """
        self.discover_and_register()
        for plugin in self.generators:
            try:
                code = plugin.generate(gen_context, context, on_chunk=on_chunk)
                if code is not None:
                    return code
            except Exception as e:
                logger.error(
                    "Generator plugin %s failed: %s", plugin.__class__.__name__, e
                )
        return None

    def run_subtask_post_processors(
        self, subtask: Any, code: str, context: PluginContext
    ) -> str:
        """Run all active post-processors on subtask code."""
        self.discover_and_register()
        current_code = code
        for plugin in self.post_processors:
            if hasattr(plugin, "post_process_subtask"):
                try:
                    current_code = plugin.post_process_subtask(
                        subtask, current_code, context
                    )
                except Exception as e:
                    logger.error(
                        "Post-processor plugin %s failed on subtask: %s",
                        plugin.__class__.__name__,
                        e,
                    )
        return current_code

    def run_artifact_post_processors(
        self, spec: Any, code: str | dict[str, str], context: PluginContext
    ) -> str | dict[str, str]:
        """Run all active post-processors on the final integrated artifact."""
        self.discover_and_register()
        current_code = code
        for plugin in self.post_processors:
            if hasattr(plugin, "post_process_artifact"):
                try:
                    current_code = plugin.post_process_artifact(
                        spec, current_code, context
                    )
                except Exception as e:
                    logger.error(
                        "Post-processor plugin %s failed on artifact: %s",
                        plugin.__class__.__name__,
                        e,
                    )
        return current_code

    def run_verifiers(
        self, subtask: Any, code: str | dict[str, str], context: PluginContext
    ) -> List[PluginVerificationResult]:
        """Run all active verifier plugins."""
        self.discover_and_register()
        results = []
        for plugin in self.verifiers:
            try:
                res = plugin.verify(subtask, code, context)
                if res is not None:
                    results.append(res)
            except Exception as e:
                logger.error(
                    "Verifier plugin %s failed: %s", plugin.__class__.__name__, e
                )
        return results

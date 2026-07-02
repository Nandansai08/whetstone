from builder_agent.plugin_system.base import (
    GenerationContext,
    GeneratorPlugin,
    PluginContext,
    PluginVerificationResult,
    PostProcessorPlugin,
    VerifierPlugin,
)
from builder_agent.plugin_system.manager import PluginManager

registered_plugins = []


def register_verifier(cls):
    """Decorator to register a custom Verifier plugin class."""
    if cls not in registered_plugins:
        registered_plugins.append(cls)
    return cls


def register_generator(cls):
    """Decorator to register a custom Generator plugin class."""
    if cls not in registered_plugins:
        registered_plugins.append(cls)
    return cls


def register_post_processor(cls):
    """Decorator to register a custom PostProcessor plugin class."""
    if cls not in registered_plugins:
        registered_plugins.append(cls)
    return cls


__all__ = [
    "PluginManager",
    "PluginContext",
    "GenerationContext",
    "PluginVerificationResult",
    "VerifierPlugin",
    "GeneratorPlugin",
    "PostProcessorPlugin",
    "register_verifier",
    "register_generator",
    "register_post_processor",
]

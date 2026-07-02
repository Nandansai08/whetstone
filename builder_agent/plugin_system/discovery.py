import importlib.metadata
import importlib.util
import logging
import pathlib
import sys
from typing import Any, List

logger = logging.getLogger(__name__)


def load_entry_point_plugins() -> List[Any]:
    """Query and load plugin classes from Python entry points.

    Queries entry points registered under the 'whetstone.plugins' group.
    """
    plugins = []
    try:
        eps = importlib.metadata.entry_points(group="whetstone.plugins")
        for ep in eps:
            try:
                plugin_cls = ep.load()
                plugins.append(plugin_cls)
            except Exception as e:
                logger.error("Failed to load entry point plugin %s: %s", ep.name, e)
    except Exception as e:
        logger.error("Failed to query entry point plugins: %s", e)
    return plugins


def load_local_plugins(plugin_dir: str = "plugins") -> List[Any]:
    """Scan and dynamically load plugin classes from a local plugins directory."""
    plugins = []
    plugins_path = pathlib.Path.cwd() / plugin_dir
    if not plugins_path.exists() or not plugins_path.is_dir():
        return plugins

    if str(plugins_path) not in sys.path:
        sys.path.insert(0, str(plugins_path))

    for path in plugins_path.glob("*.py"):
        if path.name == "__init__.py":
            continue
        module_name = f"whetstone_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Scan classes in the loaded module
                for name, attr in vars(module).items():
                    if isinstance(attr, type) and attr.__module__ == module_name:
                        # Check if the class implements any plugin interface methods
                        if (
                            hasattr(attr, "verify")
                            or hasattr(attr, "generate")
                            or hasattr(attr, "post_process_subtask")
                            or hasattr(attr, "post_process_artifact")
                        ):
                            plugins.append(attr)
        except Exception as e:
            logger.error("Failed to load local plugin from %s: %s", path, e)

    return plugins

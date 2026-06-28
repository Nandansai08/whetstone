import os


def is_safe_relative_path(path: str) -> bool:
    """Validate that the path is relative and does not permit directory traversal."""
    # Reject absolute paths (starting with / or matching drive letter on Windows)
    if os.path.isabs(path) or path.startswith(("/", "\\")):
        return False
    # Reject drive letter (e.g. C:) or special characters
    if ":" in path:
        return False
    # Reject path traversal (e.g. contain ..)
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part == "..":
            return False
    return True


def validate_and_resolve_path(base_dir: str, rel_path: str) -> str:
    """
    Resolves base_dir and rel_path using os.path.realpath() and verifies
    that the resolved path is strictly contained within base_dir.
    Raises ValueError if unsafe path traversal or symlink escape is detected.
    """
    if not is_safe_relative_path(rel_path):
        raise ValueError(f"Unsafe path traversal/absolute path detected: {rel_path}")

    real_base = os.path.realpath(base_dir)
    joined_path = os.path.join(real_base, rel_path)
    real_dest = os.path.realpath(joined_path)

    try:
        common = os.path.commonpath([real_base, real_dest])
    except ValueError:
        raise ValueError(
            f"Unsafe path traversal detected: {rel_path} escapes {base_dir}"
        )

    if common != real_base:
        raise ValueError(
            f"Unsafe path traversal detected: {rel_path} escapes {base_dir}"
        )

    return real_dest

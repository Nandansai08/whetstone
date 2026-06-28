import os
import tempfile

import pytest

from builder_agent.safety import is_safe_relative_path, validate_and_resolve_path


def test_is_safe_relative_path():
    # Safe paths
    assert is_safe_relative_path("foo/bar.py") is True
    assert is_safe_relative_path("foo/bar/baz") is True
    assert is_safe_relative_path("a.py") is True

    # Absolute paths
    assert is_safe_relative_path("/foo/bar.py") is False
    assert is_safe_relative_path("C:/foo/bar.py") is False
    assert is_safe_relative_path("\\foo\\bar.py") is False

    # Traversal paths
    assert is_safe_relative_path("../foo.py") is False
    assert is_safe_relative_path("foo/../bar.py") is False
    assert is_safe_relative_path("foo/bar/..") is False
    assert is_safe_relative_path("foo:bar.py") is False

def test_validate_and_resolve_path_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = validate_and_resolve_path(tmpdir, "foo/bar.py")
        real_tmp = os.path.realpath(tmpdir)
        assert dest == os.path.join(real_tmp, "foo", "bar.py")

def test_validate_and_resolve_path_traversal_rejection():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unsafe path traversal"):
            validate_and_resolve_path(tmpdir, "../unsafe.py")
        with pytest.raises(ValueError, match="Unsafe path traversal"):
            validate_and_resolve_path(tmpdir, "foo/../../unsafe.py")

def test_validate_and_resolve_path_symlink_escape():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory outside base_dir
        with tempfile.TemporaryDirectory() as outside_dir:
            real_base = os.path.realpath(tmpdir)
            real_outside = os.path.realpath(outside_dir)

            # Create a subdirectory inside base_dir
            sub_dir = os.path.join(real_base, "sub")
            os.makedirs(sub_dir, exist_ok=True)

            link_path = os.path.join(sub_dir, "escapelink")

            # Attempt to create symlink pointing to the outside directory
            try:
                os.symlink(real_outside, link_path)
            except (OSError, NotImplementedError):
                pytest.skip(
                    "Symlinks are not supported or permissions prevent creating them"
                )

            # Now, attempt to write to target inside the symlink (which escapes base)
            with pytest.raises(ValueError, match="Unsafe path traversal"):
                validate_and_resolve_path(real_base, "sub/escapelink/malicious.py")

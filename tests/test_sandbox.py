import pytest
from pathlib import Path
from src.llm import set_allowed_paths, get_allowed_paths, is_path_allowed, _execute_local_tool


def test_allowed_paths_thread_local():
    set_allowed_paths(["/allowed/path1", "/allowed/path2"])
    assert get_allowed_paths() == ["/allowed/path1", "/allowed/path2"]
    set_allowed_paths(None)
    assert get_allowed_paths() is None


def test_is_path_allowed():
    allowed = ["/allowed/dir1", "/home/user/workspace/trial_123"]

    # Exact matches
    assert is_path_allowed("/allowed/dir1", allowed)
    assert is_path_allowed("/home/user/workspace/trial_123", allowed)

    # Subpaths (files or directories inside allowed directories)
    assert is_path_allowed("/allowed/dir1/file.txt", allowed)
    assert is_path_allowed("/allowed/dir1/subdir/another_file.sv", allowed)
    assert is_path_allowed("/home/user/workspace/trial_123/rtl/module.v", allowed)

    # Relative path variations (resolved to absolute)
    assert is_path_allowed("/allowed/dir1/../dir1/file.txt", allowed)

    # Outside paths
    assert not is_path_allowed("/allowed/dir2", allowed)
    assert not is_path_allowed("/home/user/workspace/trial_123_suffix", allowed)
    assert not is_path_allowed("/etc/passwd", allowed)
    assert not is_path_allowed("/", allowed)


def test_execute_local_tool_sandbox(tmp_path):
    allowed_dir = tmp_path / "sandbox"
    outside_dir = tmp_path / "outside"
    allowed_dir.mkdir()
    outside_dir.mkdir()

    allowed_file = allowed_dir / "test.txt"
    outside_file = outside_dir / "secret.txt"

    allowed_file.write_text("Hello from sandbox", encoding="utf-8")
    outside_file.write_text("Secret outside", encoding="utf-8")

    # When no restrictions are active, both paths can be accessed
    set_allowed_paths(None)
    assert "Hello from sandbox" in _execute_local_tool("read_file", {"path": str(allowed_file)})
    assert "Secret outside" in _execute_local_tool("read_file", {"path": str(outside_file)})

    # When sandbox is active, only the allowed dir can be accessed
    set_allowed_paths([str(allowed_dir)])
    try:
        # Allowed path reads successfully
        assert "Hello from sandbox" in _execute_local_tool("read_file", {"path": str(allowed_file)})

        # Outside path gets blocked with permission denied
        res = _execute_local_tool("read_file", {"path": str(outside_file)})
        assert "Permission denied" in res

        # Grep search inside sandbox works
        grep_res = _execute_local_tool("grep_search", {"pattern": "Hello", "path": str(allowed_dir)})
        assert "Permission denied" not in grep_res

        # Grep search outside gets blocked
        grep_outside = _execute_local_tool("grep_search", {"pattern": "Secret", "path": str(outside_dir)})
        assert "Permission denied" in grep_outside

        # Edit file inside sandbox works
        edit_file = allowed_dir / "new_file.txt"
        edit_res = _execute_local_tool("edit_file", {"path": str(edit_file), "content": "sandbox edit"})
        assert "Wrote" in edit_res
        assert edit_file.read_text() == "sandbox edit"

        # Edit file outside sandbox gets blocked
        edit_outside = outside_dir / "new_file.txt"
        edit_outside_res = _execute_local_tool("edit_file", {"path": str(edit_outside), "content": "outside edit"})
        assert "Permission denied" in edit_outside_res
        assert not edit_outside.exists()
    finally:
        set_allowed_paths(None)

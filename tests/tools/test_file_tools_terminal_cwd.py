"""Tests for TERMINAL_CWD-aware path resolution in file_tools and file_operations."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from tools.file_tools import _resolve_path


class TestResolvePath:
    """Unit tests for _resolve_path helper."""

    def test_absolute_path_ignores_terminal_cwd(self, tmp_path):
        with mock.patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}):
            result = _resolve_path("/usr/bin/python")
            assert result == Path("/usr/bin/python").resolve()

    def test_relative_path_uses_terminal_cwd(self, tmp_path):
        target = tmp_path / "hello.txt"
        target.touch()
        with mock.patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}):
            result = _resolve_path("hello.txt")
            assert result == target.resolve()

    def test_relative_path_falls_back_to_cwd_without_terminal_cwd(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TERMINAL_CWD", None)
            result = _resolve_path("somefile.txt")
            assert result == (Path.cwd() / "somefile.txt").resolve()

    def test_tilde_expansion(self, tmp_path):
        with mock.patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}):
            result = _resolve_path("~/test.txt")
            assert str(result).startswith(str(Path.home()))

    def test_empty_terminal_cwd_ignored(self):
        with mock.patch.dict(os.environ, {"TERMINAL_CWD": "  "}):
            result = _resolve_path("file.txt")
            assert result == (Path.cwd() / "file.txt").resolve()


class TestFileOperationsExpandPath:
    """Test that ShellFileOperations._expand_path honours TERMINAL_CWD."""

    def test_relative_path_resolved_via_terminal_cwd(self, tmp_path):
        from tools.file_operations import ShellFileOperations

        ops = ShellFileOperations.__new__(ShellFileOperations)
        # Minimal stub so _expand_path works without a real terminal
        with mock.patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}):
            result = ops._expand_path("subdir/file.py")
            assert result == os.path.join(str(tmp_path), "subdir/file.py")

    def test_absolute_path_unchanged(self, tmp_path):
        from tools.file_operations import ShellFileOperations

        ops = ShellFileOperations.__new__(ShellFileOperations)
        with mock.patch.dict(os.environ, {"TERMINAL_CWD": str(tmp_path)}):
            result = ops._expand_path("/absolute/path.py")
            assert result == "/absolute/path.py"

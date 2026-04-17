"""Tests for color contrast and NO_COLOR support (issue #11300).

Ensures the TUI does not use bright yellow (#ffff00) which is unreadable
on light terminal backgrounds, and that NO_COLOR is respected.
"""

import os
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_yellow_references(*paths: str) -> list[str]:
    """Return lines containing problematic bright-yellow Rich markup."""
    import re
    hits = []
    for path in paths:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                # Match Rich markup [yellow], [bold yellow], [dim yellow]
                # but ignore comments and string literals about the word "yellow"
                if re.search(r'\[(bold |dim )?yellow\]', line):
                    hits.append(f"{path}:{i}: {line.rstrip()}")
    return hits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoBrightYellowMarkup:
    """No Rich [yellow] markup should remain in the TUI source files."""

    SOURCE_FILES = [
        "hermes_cli/banner.py",
        "hermes_cli/skills_hub.py",
        "hermes_cli/plugins_cmd.py",
        "cli.py",
    ]

    def test_no_yellow_rich_markup(self):
        hits = _collect_yellow_references(*self.SOURCE_FILES)
        assert hits == [], (
            "Found bright-yellow Rich markup (unreadable on light backgrounds):\n"
            + "\n".join(hits)
        )


class TestNoColorEnvVar:
    """NO_COLOR environment variable must suppress color output."""

    def test_colors_module_respects_no_color(self):
        from hermes_cli.colors import should_use_color
        old = os.environ.get("NO_COLOR")
        try:
            os.environ["NO_COLOR"] = ""
            assert should_use_color() is False
        finally:
            if old is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = old

    def test_chat_console_respects_no_color(self):
        """ChatConsole must disable color when NO_COLOR is set."""
        old = os.environ.get("NO_COLOR")
        try:
            os.environ["NO_COLOR"] = "1"
            # ChatConsole reads NO_COLOR at construction time
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from cli import ChatConsole
            cc = ChatConsole()
            assert cc._inner.no_color is True
        finally:
            if old is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = old

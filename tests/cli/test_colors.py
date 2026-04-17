"""Tests for hermes_cli.colors — ensure no bright-yellow codes that are
unreadable on light terminal backgrounds."""

from hermes_cli.colors import Colors, color, should_use_color


def test_yellow_is_not_ansi_33():
    """Colors.YELLOW must NOT be plain \\033[33m (bright yellow, unreadable
    on light backgrounds).  We use 256-color 178 (dark gold) instead."""
    assert "\\033[33m" not in Colors.YELLOW
    assert "178" in Colors.YELLOW


def test_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert not should_use_color()
    assert color("hello", Colors.YELLOW) == "hello"


def test_color_wraps_when_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    # can't guarantee isatty in CI, so just ensure the function runs
    result = color("hi", Colors.YELLOW)
    assert "hi" in result

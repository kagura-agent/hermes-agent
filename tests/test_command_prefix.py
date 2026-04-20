"""Tests for configurable command_prefix (issue #12688)."""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType


# ── MessageEvent with default prefix ──

class TestMessageEventDefaultPrefix:
    def test_is_command_slash(self):
        e = MessageEvent(text="/help")
        assert e.is_command()

    def test_is_command_plain_text(self):
        e = MessageEvent(text="hello")
        assert not e.is_command()

    def test_get_command(self):
        e = MessageEvent(text="/reset some args")
        assert e.get_command() == "reset"

    def test_get_command_args(self):
        e = MessageEvent(text="/model gpt-4")
        assert e.get_command_args() == "gpt-4"

    def test_get_command_args_empty(self):
        e = MessageEvent(text="/help")
        assert e.get_command_args() == ""

    def test_not_command_returns_full_text(self):
        e = MessageEvent(text="just text")
        assert e.get_command_args() == "just text"


# ── MessageEvent with custom prefix ──

class TestMessageEventCustomPrefix:
    def test_is_command_bang(self):
        e = MessageEvent(text="!help", command_prefix="!")
        assert e.is_command()

    def test_slash_not_command_with_bang_prefix(self):
        e = MessageEvent(text="/help", command_prefix="!")
        assert not e.is_command()

    def test_get_command_bang(self):
        e = MessageEvent(text="!reset args", command_prefix="!")
        assert e.get_command() == "reset"

    def test_get_command_args_bang(self):
        e = MessageEvent(text="!model gpt-4", command_prefix="!")
        assert e.get_command_args() == "gpt-4"

    def test_backslash_prefix(self):
        e = MessageEvent(text="\\help", command_prefix="\\")
        assert e.is_command()
        assert e.get_command() == "help"

    def test_multi_char_prefix(self):
        e = MessageEvent(text="!!help", command_prefix="!!")
        assert e.is_command()
        assert e.get_command() == "help"


# ── GatewayConfig.get_command_prefix ──

class TestGatewayConfigCommandPrefix:
    def test_default_prefix(self):
        cfg = GatewayConfig()
        assert cfg.get_command_prefix() == "/"

    def test_global_override(self):
        cfg = GatewayConfig(command_prefix="!")
        assert cfg.get_command_prefix() == "!"
        assert cfg.get_command_prefix(Platform.SLACK) == "!"

    def test_platform_override(self):
        cfg = GatewayConfig(
            command_prefix="/",
            platform_command_prefix={Platform.MATRIX: "\\", Platform.MATTERMOST: "!"},
        )
        assert cfg.get_command_prefix() == "/"
        assert cfg.get_command_prefix(Platform.MATRIX) == "\\"
        assert cfg.get_command_prefix(Platform.MATTERMOST) == "!"
        assert cfg.get_command_prefix(Platform.SLACK) == "/"

    def test_from_dict_command_prefix(self):
        cfg = GatewayConfig.from_dict({
            "command_prefix": "!",
            "platform_command_prefix": {"matrix": "\\"},
        })
        assert cfg.command_prefix == "!"
        assert cfg.platform_command_prefix == {Platform.MATRIX: "\\"}
        assert cfg.get_command_prefix(Platform.MATRIX) == "\\"
        assert cfg.get_command_prefix(Platform.DISCORD) == "!"

    def test_from_dict_defaults(self):
        cfg = GatewayConfig.from_dict({})
        assert cfg.command_prefix == "/"
        assert cfg.platform_command_prefix == {}

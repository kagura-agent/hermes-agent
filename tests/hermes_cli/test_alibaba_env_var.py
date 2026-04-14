"""Tests for alibaba provider env var handling (issue #9506)."""

import os
import sys
import types

import pytest

# Ensure dotenv doesn't interfere
if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.auth import PROVIDER_REGISTRY


class TestAlibabaEnvVars:
    """Verify the alibaba provider accepts both DASHSCOPE_API_KEY and ALIBABA_API_KEY."""

    def test_alibaba_registered(self):
        assert "alibaba" in PROVIDER_REGISTRY

    def test_dashscope_is_primary(self):
        pconfig = PROVIDER_REGISTRY["alibaba"]
        assert pconfig.api_key_env_vars[0] == "DASHSCOPE_API_KEY"

    def test_alibaba_api_key_is_fallback(self):
        pconfig = PROVIDER_REGISTRY["alibaba"]
        assert "ALIBABA_API_KEY" in pconfig.api_key_env_vars

    def test_env_var_order(self):
        pconfig = PROVIDER_REGISTRY["alibaba"]
        assert pconfig.api_key_env_vars == ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY")


class TestAlibabaErrorMessage:
    """Verify error messages show actual env var names from the registry."""

    def test_error_message_shows_dashscope(self):
        """The error hint for alibaba must mention DASHSCOPE_API_KEY, not ALIBABA_API_KEY."""
        pconfig = PROVIDER_REGISTRY.get("alibaba")
        assert pconfig and pconfig.api_key_env_vars
        env_hint = ' or '.join(pconfig.api_key_env_vars)
        assert "DASHSCOPE_API_KEY" in env_hint

    def test_error_message_does_not_guess_from_name(self):
        """The old code used {provider.upper()}_API_KEY which gives ALIBABA_API_KEY.
        The new code should use the registry, so DASHSCOPE_API_KEY comes first."""
        pconfig = PROVIDER_REGISTRY["alibaba"]
        env_hint = ' or '.join(pconfig.api_key_env_vars)
        # The hint should start with DASHSCOPE_API_KEY, not ALIBABA_API_KEY
        assert env_hint.startswith("DASHSCOPE_API_KEY")

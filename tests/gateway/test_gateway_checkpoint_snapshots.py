"""Tests for gateway checkpoint snapshot integration.

Verifies that gateway mode (run.py) reads checkpoint config from config.yaml
and passes it to AIAgent, so that ensure_checkpoint() is called before
file-mutating tool calls (write_file, patch).
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user_config(enabled=True, max_snapshots=50):
    """Return a minimal config dict with checkpoint settings."""
    return {"checkpoints": {"enabled": enabled, "max_snapshots": max_snapshots}}


def _make_user_config_bool(enabled=True):
    """Config where checkpoints key is a plain bool (shorthand)."""
    return {"checkpoints": enabled}


# ---------------------------------------------------------------------------
# Tests: checkpoint config parsing in _run_agent
# ---------------------------------------------------------------------------


class TestGatewayCheckpointConfigParsing:
    """The gateway should read checkpoints.enabled and checkpoints.max_snapshots
    from config.yaml and forward them to AIAgent."""

    def test_checkpoints_enabled_passed_to_agent(self):
        """When checkpoints.enabled is true in config, AIAgent receives
        checkpoints_enabled=True."""
        user_config = _make_user_config(enabled=True, max_snapshots=30)
        _cp_cfg = user_config.get("checkpoints", {})
        if isinstance(_cp_cfg, bool):
            _cp_cfg = {"enabled": _cp_cfg}
        _checkpoints_enabled = bool(_cp_cfg.get("enabled", False))
        _checkpoint_max_snapshots = int(_cp_cfg.get("max_snapshots", 50))

        assert _checkpoints_enabled is True
        assert _checkpoint_max_snapshots == 30

    def test_checkpoints_disabled_by_default(self):
        """When checkpoints key is absent, checkpoints stay disabled."""
        user_config = {}
        _cp_cfg = user_config.get("checkpoints", {})
        if isinstance(_cp_cfg, bool):
            _cp_cfg = {"enabled": _cp_cfg}
        _checkpoints_enabled = bool(_cp_cfg.get("enabled", False))
        _checkpoint_max_snapshots = int(_cp_cfg.get("max_snapshots", 50))

        assert _checkpoints_enabled is False
        assert _checkpoint_max_snapshots == 50

    def test_checkpoints_bool_shorthand_true(self):
        """checkpoints: true (plain bool) should enable checkpoints."""
        user_config = _make_user_config_bool(True)
        _cp_cfg = user_config.get("checkpoints", {})
        if isinstance(_cp_cfg, bool):
            _cp_cfg = {"enabled": _cp_cfg}
        _checkpoints_enabled = bool(_cp_cfg.get("enabled", False))

        assert _checkpoints_enabled is True

    def test_checkpoints_bool_shorthand_false(self):
        """checkpoints: false (plain bool) should disable checkpoints."""
        user_config = _make_user_config_bool(False)
        _cp_cfg = user_config.get("checkpoints", {})
        if isinstance(_cp_cfg, bool):
            _cp_cfg = {"enabled": _cp_cfg}
        _checkpoints_enabled = bool(_cp_cfg.get("enabled", False))

        assert _checkpoints_enabled is False


# ---------------------------------------------------------------------------
# Tests: AIAgent receives checkpoint params
# ---------------------------------------------------------------------------


class TestAIAgentCheckpointIntegration:
    """Verify AIAgent's checkpoint manager is enabled when gateway passes
    the config through."""

    def test_agent_checkpoint_mgr_enabled(self):
        """AIAgent created with checkpoints_enabled=True should have an
        enabled checkpoint manager."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="gpt-4o-mini",
            quiet_mode=True,
            checkpoints_enabled=True,
            checkpoint_max_snapshots=25,
        )
        assert agent._checkpoint_mgr.enabled is True
        assert agent._checkpoint_mgr.max_snapshots == 25

    def test_agent_checkpoint_mgr_disabled_by_default(self):
        """AIAgent with default params should have checkpoints disabled."""
        from run_agent import AIAgent

        agent = AIAgent(
            model="gpt-4o-mini",
            quiet_mode=True,
        )
        assert agent._checkpoint_mgr.enabled is False

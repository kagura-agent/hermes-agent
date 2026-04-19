"""Tests for terminal.backend display in TUI session info and config.show."""

import sys
from unittest.mock import MagicMock, patch

import pytest

_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(get_hermes_home=MagicMock(return_value="/tmp/hermes_test")),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()
        mod._methods.clear()
        importlib.reload(mod)


def _make_agent():
    agent = MagicMock()
    agent.model = "test-model"
    agent.tools = []
    agent.context_compressor = None
    return agent


# ── _session_info includes terminal_backend ─────────────────────────


def test_session_info_reads_terminal_backend_from_config(server):
    agent = _make_agent()
    with patch.object(server, "_load_cfg", return_value={"terminal": {"backend": "local"}}):
        info = server._session_info(agent)
    assert info["terminal_backend"] == "local"


def test_session_info_terminal_backend_defaults_to_local(server):
    agent = _make_agent()
    with patch.object(server, "_load_cfg", return_value={}):
        info = server._session_info(agent)
    assert info["terminal_backend"] == "local"


def test_session_info_terminal_backend_docker(server):
    agent = _make_agent()
    with patch.object(server, "_load_cfg", return_value={"terminal": {"backend": "docker"}}):
        info = server._session_info(agent)
    assert info["terminal_backend"] == "docker"


# ── config.show includes Terminal section ────────────────────────────


def test_config_show_includes_terminal_section(server):
    from pathlib import Path
    with patch.object(server, "_load_cfg", return_value={"terminal": {"backend": "local"}}), \
         patch.object(server, "_resolve_model", return_value="test-model"), \
         patch.object(server, "_hermes_home", Path("/tmp/hermes_test")):
        resp = server.handle_request({"id": "r1", "method": "config.show", "params": {}})

    sections = resp["result"]["sections"]
    terminal_sections = [s for s in sections if s["title"] == "Terminal"]
    assert len(terminal_sections) == 1
    assert ["Backend", "local"] in terminal_sections[0]["rows"]


def test_config_show_terminal_backend_defaults_to_local(server):
    from pathlib import Path
    with patch.object(server, "_load_cfg", return_value={}), \
         patch.object(server, "_resolve_model", return_value="test-model"), \
         patch.object(server, "_hermes_home", Path("/tmp/hermes_test")):
        resp = server.handle_request({"id": "r1", "method": "config.show", "params": {}})

    sections = resp["result"]["sections"]
    terminal_sections = [s for s in sections if s["title"] == "Terminal"]
    assert len(terminal_sections) == 1
    assert ["Backend", "local"] in terminal_sections[0]["rows"]

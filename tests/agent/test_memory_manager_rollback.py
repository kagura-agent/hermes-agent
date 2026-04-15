"""Tests for add_provider() rollback on schema load failure (#9948).

Verifies that a broken external provider that raises in get_tool_schemas()
does NOT leave MemoryManager in a half-registered (poisoned) state.
"""

import json
import pytest

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Test providers
# ---------------------------------------------------------------------------

class _StableProvider(MemoryProvider):
    """Minimal working provider."""

    def __init__(self, name="stable", tools=None):
        self._name = name
        self._tools = tools or []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return self._tools

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"handled": tool_name})


class _BrokenSchemaProvider(MemoryProvider):
    """Provider whose get_tool_schemas() always raises."""

    def __init__(self, name="broken"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        raise RuntimeError("schema load exploded")

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"handled": tool_name})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAddProviderRollback:
    """Regression tests for #9948: add_provider must not mutate state if
    get_tool_schemas() raises."""

    def test_broken_provider_does_not_register(self):
        """A provider that fails in get_tool_schemas() must not appear in
        _providers, _has_external, or _tool_to_provider."""
        mgr = MemoryManager()
        builtin = _StableProvider("builtin")
        mgr.add_provider(builtin)

        broken = _BrokenSchemaProvider("broken_ext")

        with pytest.raises(RuntimeError, match="schema load exploded"):
            mgr.add_provider(broken)

        # State must be clean — no trace of the broken provider
        assert [p.name for p in mgr.providers] == ["builtin"]
        assert mgr._has_external is False
        assert mgr.get_all_tool_names() == set()

    def test_valid_provider_succeeds_after_broken_one(self):
        """After a failed add, a subsequent valid external provider must
        register successfully — _has_external must not be stuck True."""
        mgr = MemoryManager()
        builtin = _StableProvider("builtin")
        mgr.add_provider(builtin)

        broken = _BrokenSchemaProvider("broken_ext")
        with pytest.raises(RuntimeError):
            mgr.add_provider(broken)

        good = _StableProvider("good_ext", tools=[
            {"name": "ext_tool", "description": "Works", "parameters": {}},
        ])
        mgr.add_provider(good)

        assert [p.name for p in mgr.providers] == ["builtin", "good_ext"]
        assert mgr._has_external is True
        assert mgr.has_tool("ext_tool")

    def test_normal_add_provider_still_works(self):
        """Sanity check: the happy path is unaffected by the fix."""
        mgr = MemoryManager()
        builtin = _StableProvider("builtin", tools=[
            {"name": "mem_read", "description": "Read", "parameters": {}},
        ])
        ext = _StableProvider("external", tools=[
            {"name": "ext_recall", "description": "Recall", "parameters": {}},
        ])
        mgr.add_provider(builtin)
        mgr.add_provider(ext)

        assert [p.name for p in mgr.providers] == ["builtin", "external"]
        assert mgr._has_external is True
        assert mgr.has_tool("mem_read")
        assert mgr.has_tool("ext_recall")
        result = json.loads(mgr.handle_tool_call("ext_recall", {}))
        assert result["handled"] == "ext_recall"

    def test_broken_builtin_does_not_register(self):
        """Even a builtin provider must not half-register on schema failure."""
        mgr = MemoryManager()
        broken_builtin = _BrokenSchemaProvider("builtin")

        with pytest.raises(RuntimeError):
            mgr.add_provider(broken_builtin)

        assert mgr.providers == []
        assert mgr.get_all_tool_names() == set()

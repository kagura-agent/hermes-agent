"""Tests for Hindsight circuit breaker behaviour."""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    _CIRCUIT_BREAKER_COOLDOWN,
    _CIRCUIT_BREAKER_THRESHOLD,
)


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    """Create a local_embedded provider with a mock client."""
    config = {
        "mode": "local_embedded",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("plugins.memory.hindsight._check_local_runtime", lambda: (True, ""))

    p = HindsightMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")

    client = MagicMock()
    client.aretain = AsyncMock(return_value=SimpleNamespace(ok=True))
    client.arecall = AsyncMock(
        return_value=SimpleNamespace(results=[SimpleNamespace(text="m1")])
    )
    client.aretain_batch = AsyncMock()
    client.aclose = AsyncMock()
    p._client = client
    return p


class TestCircuitBreaker:

    def test_opens_after_threshold_failures(self, provider):
        # Use a non-retriable error so each call increments the counter directly
        provider._client.arecall = AsyncMock(side_effect=RuntimeError("database migration failed"))

        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            with pytest.raises(RuntimeError, match="database migration failed"):
                provider._run_hindsight_operation(lambda c: c.arecall(bank_id="b", query="q"))

        with pytest.raises(RuntimeError, match="circuit breaker open"):
            provider._run_hindsight_operation(lambda c: c.arecall(bank_id="b", query="q"))

    def test_resets_after_cooldown(self, provider):
        provider._circuit_breaker_failures = _CIRCUIT_BREAKER_THRESHOLD
        provider._circuit_breaker_open_until = time.monotonic() - 1  # expired

        result = provider._run_hindsight_operation(lambda c: c.arecall(bank_id="b", query="q"))
        assert result.results
        assert provider._circuit_breaker_failures == 0

    def test_success_resets_counter(self, provider):
        provider._circuit_breaker_failures = 2

        provider._run_hindsight_operation(lambda c: c.arecall(bank_id="b", query="q"))
        assert provider._circuit_breaker_failures == 0

    def test_sync_turn_skips_when_circuit_open(self, provider):
        provider._circuit_breaker_failures = _CIRCUIT_BREAKER_THRESHOLD
        provider._circuit_breaker_open_until = time.monotonic() + 300
        provider._auto_retain = True
        provider._retain_every_n_turns = 1

        provider.sync_turn("hello", "world")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)

        provider._client.aretain_batch.assert_not_called()

    def test_retriable_errors_include_daemon_markers(self, provider):
        assert provider._is_retriable_embedded_connection_error(
            RuntimeError("Failed to start daemon on port 9999")
        )
        assert provider._is_retriable_embedded_connection_error(
            RuntimeError("Cannot use HindsightEmbedded after it has been closed")
        )

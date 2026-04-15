"""Tests for watch_match thread-aware routing (#10411).

Ensures that background watch_match notifications are only injected into
the originating thread/session, not whichever thread happens to be active
when the completion_queue is drained.
"""

import queue
import time
import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def registry():
    return ProcessRegistry()


def _make_session(
    sid="proc_routing",
    command="tail -f app.log",
    session_key="agent:main:discord:thread:chan1:thr1",
    platform="discord",
    chat_id="chan1",
    thread_id="thr1",
    user_id="u1",
    user_name="alice",
    watch_patterns=None,
) -> ProcessSession:
    s = ProcessSession(
        id=sid,
        command=command,
        session_key=session_key,
        started_at=time.time(),
        watch_patterns=watch_patterns or ["ERROR"],
        watcher_platform=platform,
        watcher_chat_id=chat_id,
        watcher_thread_id=thread_id,
        watcher_user_id=user_id,
        watcher_user_name=user_name,
    )
    return s


# =========================================================================
# watch_match events carry routing metadata
# =========================================================================

class TestWatchMatchRoutingMetadata:
    """Verify watch_match events include all fields needed for routing."""

    def test_watch_match_includes_session_key(self, registry):
        session = _make_session()
        registry._check_watch_patterns(session, "ERROR: disk full\n")
        evt = registry.completion_queue.get_nowait()
        assert evt["type"] == "watch_match"
        assert evt["session_key"] == "agent:main:discord:thread:chan1:thr1"

    def test_watch_match_includes_platform(self, registry):
        session = _make_session()
        registry._check_watch_patterns(session, "ERROR: crash\n")
        evt = registry.completion_queue.get_nowait()
        assert evt["platform"] == "discord"
        assert evt["chat_id"] == "chan1"
        assert evt["thread_id"] == "thr1"

    def test_watch_match_includes_user_info(self, registry):
        session = _make_session()
        registry._check_watch_patterns(session, "ERROR: fail\n")
        evt = registry.completion_queue.get_nowait()
        assert evt["user_id"] == "u1"
        assert evt["user_name"] == "alice"

    def test_watch_disabled_includes_routing(self, registry):
        """watch_disabled events also carry routing metadata."""
        session = _make_session()
        from tools.process_registry import WATCH_MAX_PER_WINDOW, WATCH_OVERLOAD_KILL_SECONDS
        # Fill rate window
        for i in range(WATCH_MAX_PER_WINDOW):
            registry._check_watch_patterns(session, f"ERROR {i}\n")
        # Force sustained overload
        session._watch_overload_since = time.time() - WATCH_OVERLOAD_KILL_SECONDS - 1
        registry._check_watch_patterns(session, "ERROR overload\n")
        # Find the watch_disabled event
        disabled_evt = None
        while not registry.completion_queue.empty():
            evt = registry.completion_queue.get_nowait()
            if evt.get("type") == "watch_disabled":
                disabled_evt = evt
        assert disabled_evt is not None
        assert disabled_evt["session_key"] == "agent:main:discord:thread:chan1:thr1"
        assert disabled_evt["platform"] == "discord"
        assert disabled_evt["thread_id"] == "thr1"


# =========================================================================
# Drain loop session filtering (simulated)
# =========================================================================

class TestDrainLoopFiltering:
    """
    Simulate the drain-loop logic from gateway/run.py to verify that only
    events matching the current session_key are injected, and non-matching
    events are re-queued.
    """

    @staticmethod
    def _drain_with_session_filter(completion_queue, current_session_key):
        """Replicate the drain logic from _handle_message_with_agent."""
        injected = []
        requeued = []
        while not completion_queue.empty():
            evt = completion_queue.get_nowait()
            evt_type = evt.get("type", "completion")
            if evt_type in ("watch_match", "watch_disabled"):
                evt_session_key = str(evt.get("session_key") or "").strip()
                if not evt_session_key or evt_session_key == current_session_key:
                    injected.append(evt)
                else:
                    requeued.append(evt)
        # Re-queue non-matching
        for evt in requeued:
            completion_queue.put(evt)
        return injected, requeued

    def test_matching_event_injected(self, registry):
        """Event with same session_key as current thread is injected."""
        session = _make_session(
            session_key="agent:main:discord:thread:chan1:thr1",
        )
        registry._check_watch_patterns(session, "ERROR: match\n")

        injected, requeued = self._drain_with_session_filter(
            registry.completion_queue,
            "agent:main:discord:thread:chan1:thr1",
        )
        assert len(injected) == 1
        assert len(requeued) == 0
        assert injected[0]["session_key"] == "agent:main:discord:thread:chan1:thr1"

    def test_non_matching_event_requeued(self, registry):
        """Event with different session_key is re-queued, not injected."""
        session = _make_session(
            session_key="agent:main:discord:thread:chan1:thr_other",
            thread_id="thr_other",
        )
        registry._check_watch_patterns(session, "ERROR: wrong thread\n")

        injected, requeued = self._drain_with_session_filter(
            registry.completion_queue,
            "agent:main:discord:thread:chan1:thr1",
        )
        assert len(injected) == 0
        assert len(requeued) == 1
        # Verify it's back in the queue
        assert not registry.completion_queue.empty()
        evt = registry.completion_queue.get_nowait()
        assert evt["session_key"] == "agent:main:discord:thread:chan1:thr_other"

    def test_mixed_events_correctly_partitioned(self, registry):
        """Mix of matching and non-matching events are correctly separated."""
        current_key = "agent:main:discord:thread:chan1:thr1"

        # Event for current thread
        s1 = _make_session(sid="proc_1", session_key=current_key, thread_id="thr1")
        registry._check_watch_patterns(s1, "ERROR: for me\n")

        # Event for different thread
        s2 = _make_session(
            sid="proc_2",
            session_key="agent:main:discord:thread:chan1:thr2",
            thread_id="thr2",
        )
        registry._check_watch_patterns(s2, "ERROR: for other\n")

        # Another event for current thread
        s3 = _make_session(sid="proc_3", session_key=current_key, thread_id="thr1")
        registry._check_watch_patterns(s3, "ERROR: also for me\n")

        injected, requeued = self._drain_with_session_filter(
            registry.completion_queue, current_key,
        )
        assert len(injected) == 2
        assert len(requeued) == 1
        assert all(e["session_key"] == current_key for e in injected)
        assert requeued[0]["session_key"] == "agent:main:discord:thread:chan1:thr2"

    def test_empty_session_key_event_always_injected(self, registry):
        """Event with no session_key (legacy/fallback) is always injected."""
        session = _make_session(session_key="")
        registry._check_watch_patterns(session, "ERROR: legacy\n")

        injected, requeued = self._drain_with_session_filter(
            registry.completion_queue,
            "agent:main:discord:thread:chan1:thr1",
        )
        assert len(injected) == 1
        assert len(requeued) == 0

    def test_completion_events_not_filtered(self, registry):
        """Completion events (non-watch) are neither injected nor re-queued by the watch drain."""
        registry.completion_queue.put({
            "type": "completion",
            "session_id": "proc_done",
            "session_key": "agent:main:discord:thread:chan1:thr_other",
            "command": "pytest",
            "exit_code": 0,
            "output": "ok",
        })

        injected, requeued = self._drain_with_session_filter(
            registry.completion_queue,
            "agent:main:discord:thread:chan1:thr1",
        )
        # Completion events should pass through (not be treated as watch events)
        assert len(injected) == 0
        assert len(requeued) == 0

    def test_requeued_events_delivered_to_correct_thread(self, registry):
        """Events re-queued from thread A's drain are picked up by thread B's drain."""
        key_a = "agent:main:discord:thread:chan1:thr_a"
        key_b = "agent:main:discord:thread:chan1:thr_b"

        # Event for thread B arrives while thread A is active
        s = _make_session(sid="proc_b", session_key=key_b, thread_id="thr_b")
        registry._check_watch_patterns(s, "ERROR: for B\n")

        # Thread A drains — should re-queue
        injected_a, requeued_a = self._drain_with_session_filter(
            registry.completion_queue, key_a,
        )
        assert len(injected_a) == 0
        assert len(requeued_a) == 1

        # Thread B drains — should pick it up
        injected_b, requeued_b = self._drain_with_session_filter(
            registry.completion_queue, key_b,
        )
        assert len(injected_b) == 1
        assert len(requeued_b) == 0
        assert injected_b[0]["thread_id"] == "thr_b"


# =========================================================================
# _format_gateway_process_notification
# =========================================================================

class TestFormatGatewayProcessNotification:
    def test_watch_match_format(self):
        from gateway.run import _format_gateway_process_notification
        evt = {
            "type": "watch_match",
            "session_id": "proc_1",
            "command": "tail -f log",
            "pattern": "ERROR",
            "output": "ERROR: disk full",
            "suppressed": 0,
        }
        text = _format_gateway_process_notification(evt)
        assert text is not None
        assert "proc_1" in text
        assert "ERROR" in text
        assert "disk full" in text

    def test_watch_match_with_suppressed(self):
        from gateway.run import _format_gateway_process_notification
        evt = {
            "type": "watch_match",
            "session_id": "proc_1",
            "command": "tail -f log",
            "pattern": "ERROR",
            "output": "ERROR: disk full",
            "suppressed": 3,
        }
        text = _format_gateway_process_notification(evt)
        assert "3 earlier matches" in text

    def test_watch_disabled_format(self):
        from gateway.run import _format_gateway_process_notification
        evt = {
            "type": "watch_disabled",
            "message": "Watch patterns disabled for process proc_1",
        }
        text = _format_gateway_process_notification(evt)
        assert text is not None
        assert "Watch patterns disabled" in text

    def test_completion_returns_none(self):
        from gateway.run import _format_gateway_process_notification
        evt = {"type": "completion"}
        assert _format_gateway_process_notification(evt) is None

"""Tests for gateway auto-reset parent session ID propagation (#12857).

Bug 1: get_or_create_session() must store parent_session_id in
        SessionEntry and pass it to state.db.create_session().

Bug 2: run.py must load the parent session's transcript on auto-reset
        so the agent retains prior context without manual /resume.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import (
    GatewayConfig,
    Platform,
    SessionResetPolicy,
)
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
    )


def _make_store(policy=None, tmp_path=None, db=None):
    config = GatewayConfig()
    if policy:
        config.default_reset_policy = policy
    store = SessionStore(sessions_dir=tmp_path or "/tmp/test-sessions", config=config)
    if db is not None:
        store._db = db
    return store


# ---------------------------------------------------------------------------
# Bug 1: parent_session_id propagation
# ---------------------------------------------------------------------------

class TestParentSessionIdPropagation:
    """Verify that auto-reset stores and passes parent_session_id."""

    def test_new_session_has_no_parent(self, tmp_path):
        store = _make_store(tmp_path=tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        assert entry.parent_session_id is None

    def test_auto_reset_stores_parent_session_id(self, tmp_path):
        """When idle reset triggers, the new entry must carry the old session_id."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path=tmp_path,
        )
        source = _make_source()

        # Create initial session
        first = store.get_or_create_session(source)
        first_id = first.session_id

        # Age the session so it triggers idle reset
        first.updated_at = datetime.now() - timedelta(hours=1)

        # Next access should auto-reset
        second = store.get_or_create_session(source)
        assert second.session_id != first_id
        assert second.parent_session_id == first_id
        assert second.was_auto_reset is True

    def test_parent_session_id_passed_to_db_create(self, tmp_path):
        """create_session() must receive parent_session_id kwarg."""
        mock_db = MagicMock()
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path=tmp_path,
            db=mock_db,
        )
        source = _make_source()

        first = store.get_or_create_session(source)
        first_id = first.session_id
        first.updated_at = datetime.now() - timedelta(hours=1)

        # Reset mock to only capture the second create_session call
        mock_db.reset_mock()
        second = store.get_or_create_session(source)

        # end_session should have been called for the old session
        mock_db.end_session.assert_called_once_with(first_id, "session_reset")

        # create_session must include parent_session_id
        mock_db.create_session.assert_called_once()
        call_kwargs = mock_db.create_session.call_args
        assert call_kwargs.kwargs.get("parent_session_id") == first_id or \
               (call_kwargs[1] if len(call_kwargs) > 1 else {}).get("parent_session_id") == first_id

    def test_suspended_reset_stores_parent_session_id(self, tmp_path):
        """Suspended sessions also record parent_session_id."""
        store = _make_store(tmp_path=tmp_path)
        source = _make_source()

        first = store.get_or_create_session(source)
        first_id = first.session_id
        first.suspended = True

        second = store.get_or_create_session(source)
        assert second.parent_session_id == first_id


# ---------------------------------------------------------------------------
# Bug 2: transcript carry-over on auto-reset (unit-level)
# ---------------------------------------------------------------------------

class TestAutoResetTranscriptCarryOver:
    """Verify that the parent transcript is loaded on auto-reset."""

    def test_parent_transcript_loaded_when_auto_reset(self):
        """Simulate the run.py logic: if session was auto-reset and new
        transcript is empty, parent transcript should be loaded."""
        parent_messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "session_meta", "content": "meta"},  # should be filtered
        ]

        # Simulate session_store.load_transcript behavior
        def mock_load_transcript(session_id):
            if session_id == "parent_001":
                return list(parent_messages)
            return []  # new session is empty

        session_entry = SessionEntry(
            session_key="test",
            session_id="new_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            was_auto_reset=True,
            auto_reset_reason="idle",
            parent_session_id="parent_001",
        )

        # Replicate the run.py logic
        history = mock_load_transcript(session_entry.session_id)
        if (
            getattr(session_entry, "was_auto_reset", False)
            and getattr(session_entry, "parent_session_id", None)
            and not history
        ):
            parent_history = mock_load_transcript(session_entry.parent_session_id)
            if parent_history:
                parent_history = [
                    m for m in parent_history if m.get("role") != "session_meta"
                ]
                history = parent_history

        # Should have parent messages minus session_meta
        assert len(history) == 2
        assert history[0]["content"] == "hello"
        assert history[1]["content"] == "hi there"

    def test_no_carry_over_when_new_session_has_history(self):
        """If the new session already has messages, don't load parent."""
        existing = [{"role": "user", "content": "new message"}]

        def mock_load_transcript(session_id):
            if session_id == "new_001":
                return list(existing)
            return [{"role": "user", "content": "old"}]

        session_entry = SessionEntry(
            session_key="test",
            session_id="new_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            was_auto_reset=True,
            parent_session_id="parent_001",
        )

        history = mock_load_transcript(session_entry.session_id)
        if (
            getattr(session_entry, "was_auto_reset", False)
            and getattr(session_entry, "parent_session_id", None)
            and not history
        ):
            parent_history = mock_load_transcript(session_entry.parent_session_id)
            if parent_history:
                history = parent_history

        assert len(history) == 1
        assert history[0]["content"] == "new message"

    def test_no_carry_over_without_parent_id(self):
        """If no parent_session_id, skip carry-over even on auto-reset."""
        session_entry = SessionEntry(
            session_key="test",
            session_id="new_001",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            was_auto_reset=True,
            parent_session_id=None,
        )

        history = []
        if (
            getattr(session_entry, "was_auto_reset", False)
            and getattr(session_entry, "parent_session_id", None)
            and not history
        ):
            history = [{"role": "user", "content": "should not appear"}]

        assert history == []

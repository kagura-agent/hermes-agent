"""Tests for gateway auto-reset parent session ID persistence and transcript restore.

Bug 1 (#12857): db_end_session_id was computed but never stored in SessionEntry
or passed to state.db.create_session().

Bug 2 (#12857): After auto-reset the old transcript was not restored into the
new session context (unlike CLI /resume).
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(platform=platform, chat_id=chat_id, user_id=user_id)


def _make_store(tmp_path, policy=None, db=None):
    config = GatewayConfig()
    if policy:
        config.default_reset_policy = policy
    store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db
    return store


# ---------------------------------------------------------------------------
# Bug 1: parent_session_id persisted in SessionEntry and passed to DB
# ---------------------------------------------------------------------------

class TestParentSessionIdPersistence:
    def test_auto_reset_stores_parent_session_id_in_entry(self, tmp_path):
        """After auto-reset, the new SessionEntry.parent_session_id should
        reference the old session."""
        policy = SessionResetPolicy(mode="idle", idle_minutes=1)
        store = _make_store(tmp_path, policy=policy)
        source = _make_source()

        # Create initial session
        entry1 = store.get_or_create_session(source)
        old_id = entry1.session_id
        # Simulate activity so reset_had_activity is true
        entry1.total_tokens = 100

        # Age the session past idle threshold
        entry1.updated_at = datetime.now() - timedelta(minutes=10)
        store._save()

        # Next access should auto-reset
        entry2 = store.get_or_create_session(source)
        assert entry2.session_id != old_id
        assert entry2.parent_session_id == old_id
        assert entry2.was_auto_reset is True

    def test_new_session_has_no_parent(self, tmp_path):
        """A brand-new session (no prior entry) should have parent_session_id=None."""
        store = _make_store(tmp_path)
        source = _make_source()
        entry = store.get_or_create_session(source)
        assert entry.parent_session_id is None

    def test_parent_session_id_passed_to_db_create(self, tmp_path):
        """create_session() should receive parent_session_id when auto-resetting."""
        mock_db = MagicMock()
        policy = SessionResetPolicy(mode="idle", idle_minutes=1)
        store = _make_store(tmp_path, policy=policy, db=mock_db)
        source = _make_source()

        # Create initial session
        entry1 = store.get_or_create_session(source)
        old_id = entry1.session_id
        mock_db.reset_mock()

        # Age past idle threshold
        entry1.updated_at = datetime.now() - timedelta(minutes=10)
        store._save()

        # Auto-reset
        entry2 = store.get_or_create_session(source)
        # Verify create_session was called with parent_session_id
        mock_db.create_session.assert_called_once()
        call_kwargs = mock_db.create_session.call_args[1]
        assert call_kwargs["parent_session_id"] == old_id

    def test_parent_session_id_serialization_roundtrip(self, tmp_path):
        """parent_session_id should survive to_dict/from_dict."""
        entry = SessionEntry(
            session_key="k",
            session_id="s2",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_session_id="s1",
        )
        d = entry.to_dict()
        assert d["parent_session_id"] == "s1"

        restored = SessionEntry.from_dict(d)
        assert restored.parent_session_id == "s1"

    def test_parent_session_id_none_serialization(self, tmp_path):
        """parent_session_id=None should round-trip cleanly."""
        entry = SessionEntry(
            session_key="k",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        d = entry.to_dict()
        assert d["parent_session_id"] is None
        restored = SessionEntry.from_dict(d)
        assert restored.parent_session_id is None


# ---------------------------------------------------------------------------
# Bug 2: parent transcript restored after auto-reset
# ---------------------------------------------------------------------------

class TestParentTranscriptRestore:
    """Verify that load_transcript for the parent is used when the new
    session has no history yet (auto-reset path in gateway/run.py)."""

    def test_parent_transcript_loaded_on_auto_reset(self, tmp_path):
        """Simulate the auto-reset transcript restore logic from run.py."""
        store = _make_store(tmp_path)

        parent_id = "parent_session"
        new_id = "new_session"

        # Write a transcript for the parent session
        parent_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        store.append_to_transcript(parent_id, parent_history[0])
        store.append_to_transcript(parent_id, parent_history[1])

        # Create a session entry that was auto-reset
        entry = SessionEntry(
            session_key="k",
            session_id=new_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            was_auto_reset=True,
            parent_session_id=parent_id,
        )

        # Simulate the logic from gateway/run.py
        history = store.load_transcript(entry.session_id)
        parent_sid = getattr(entry, 'parent_session_id', None)
        if parent_sid and getattr(entry, 'was_auto_reset', False) and not history:
            parent_history_loaded = store.load_transcript(parent_sid)
            if parent_history_loaded:
                history = list(parent_history_loaded)

        assert len(history) == 2
        assert history[0]["content"] == "hello"
        assert history[1]["content"] == "hi there"

    def test_no_restore_when_new_session_has_history(self, tmp_path):
        """If the new session already has its own messages, don't clobber."""
        store = _make_store(tmp_path)

        parent_id = "parent_session"
        new_id = "new_session"

        # Parent transcript
        store.append_to_transcript(parent_id, {"role": "user", "content": "old"})

        # New session already has messages
        store.append_to_transcript(new_id, {"role": "user", "content": "new msg"})

        entry = SessionEntry(
            session_key="k",
            session_id=new_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            was_auto_reset=True,
            parent_session_id=parent_id,
        )

        history = store.load_transcript(entry.session_id)
        parent_sid = getattr(entry, 'parent_session_id', None)
        if parent_sid and getattr(entry, 'was_auto_reset', False) and not history:
            history = list(store.load_transcript(parent_sid))

        # Should keep the new session's own history
        assert len(history) == 1
        assert history[0]["content"] == "new msg"

    def test_no_restore_without_auto_reset(self, tmp_path):
        """If was_auto_reset is False, don't load parent transcript."""
        store = _make_store(tmp_path)

        store.append_to_transcript("parent", {"role": "user", "content": "old"})

        entry = SessionEntry(
            session_key="k",
            session_id="new",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            was_auto_reset=False,
            parent_session_id="parent",
        )

        history = store.load_transcript(entry.session_id)
        parent_sid = getattr(entry, 'parent_session_id', None)
        if parent_sid and getattr(entry, 'was_auto_reset', False) and not history:
            history = list(store.load_transcript(parent_sid))

        assert history == []

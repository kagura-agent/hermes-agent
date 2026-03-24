"""Tests for cron execution log (memories/cron_log.md).

Verifies that:
1. Successful job runs are logged with ✅
2. Failed job runs are logged with ❌ and truncated error
3. Log file is created with header if it doesn't exist
4. Old entries are pruned when exceeding max lines
5. Logging failures are non-fatal (never crash the scheduler)
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def temp_hermes_home(tmp_path):
    """Set up a temporary HERMES_HOME for testing."""
    with patch("cron.scheduler._hermes_home", tmp_path):
        yield tmp_path


def _make_job(name="test-job", job_id="test-123"):
    return {"name": name, "id": job_id}


class TestCronExecutionLog:

    def test_successful_job_logged(self, temp_hermes_home):
        from cron.scheduler import _log_cron_execution

        _log_cron_execution(_make_job("Daily Briefing"), success=True)

        log_path = temp_hermes_home / "memories" / "cron_log.md"
        assert log_path.exists()
        content = log_path.read_text()
        assert "# Cron Execution Log" in content
        assert "✅" in content
        assert "**Daily Briefing**" in content

    def test_failed_job_logged_with_error(self, temp_hermes_home):
        from cron.scheduler import _log_cron_execution

        _log_cron_execution(
            _make_job("Backup"),
            success=False,
            error="ConnectionError: database unreachable",
        )

        log_path = temp_hermes_home / "memories" / "cron_log.md"
        content = log_path.read_text()
        assert "❌" in content
        assert "**Backup**" in content
        assert "ConnectionError" in content

    def test_error_truncated(self, temp_hermes_home):
        from cron.scheduler import _log_cron_execution

        long_error = "x" * 200
        _log_cron_execution(_make_job(), success=False, error=long_error)

        log_path = temp_hermes_home / "memories" / "cron_log.md"
        content = log_path.read_text()
        # Error should be truncated to 120 chars
        lines = [l for l in content.split("\n") if l.startswith("- [")]
        assert len(lines) == 1
        # The error portion should not contain the full 200-char string
        assert "x" * 200 not in lines[0]
        assert "x" * 120 in lines[0]

    def test_creates_memories_dir(self, temp_hermes_home):
        from cron.scheduler import _log_cron_execution

        memories_dir = temp_hermes_home / "memories"
        assert not memories_dir.exists()

        _log_cron_execution(_make_job(), success=True)
        assert memories_dir.exists()
        assert (memories_dir / "cron_log.md").exists()

    def test_appends_multiple_entries(self, temp_hermes_home):
        from cron.scheduler import _log_cron_execution

        _log_cron_execution(_make_job("Job A"), success=True)
        _log_cron_execution(_make_job("Job B"), success=True)
        _log_cron_execution(_make_job("Job C"), success=False, error="timeout")

        log_path = temp_hermes_home / "memories" / "cron_log.md"
        content = log_path.read_text()
        entries = [l for l in content.split("\n") if l.startswith("- [")]
        assert len(entries) == 3
        assert "Job A" in entries[0]
        assert "Job B" in entries[1]
        assert "Job C" in entries[2]

    def test_prunes_old_entries(self, temp_hermes_home):
        from cron.scheduler import _log_cron_execution, _CRON_LOG_MAX_LINES

        # Write more than max entries
        for i in range(_CRON_LOG_MAX_LINES + 20):
            _log_cron_execution(_make_job(f"Job-{i}"), success=True)

        log_path = temp_hermes_home / "memories" / "cron_log.md"
        content = log_path.read_text()
        entries = [l for l in content.split("\n") if l.startswith("- [")]
        assert len(entries) == _CRON_LOG_MAX_LINES
        # Should keep newest, prune oldest
        assert f"Job-{_CRON_LOG_MAX_LINES + 19}" in entries[-1]
        assert "Job-0" not in content

    def test_logging_failure_is_nonfatal(self, temp_hermes_home):
        from cron.scheduler import _log_cron_execution

        # Make memories dir unwritable
        memories_dir = temp_hermes_home / "memories"
        memories_dir.mkdir(parents=True)
        memories_dir.chmod(0o444)

        # Should not raise — just log debug and continue
        try:
            _log_cron_execution(_make_job(), success=True)
        finally:
            memories_dir.chmod(0o755)

    def test_job_without_name_uses_id(self, temp_hermes_home):
        from cron.scheduler import _log_cron_execution

        _log_cron_execution({"id": "abc-456"}, success=True)

        log_path = temp_hermes_home / "memories" / "cron_log.md"
        content = log_path.read_text()
        assert "**abc-456**" in content

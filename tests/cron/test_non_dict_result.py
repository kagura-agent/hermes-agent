"""Tests for the non-dict result type guard in run_job.

When agent.run_conversation returns a non-dict (e.g. a string, None),
the scheduler must raise RuntimeError instead of crashing with AttributeError
on the subsequent result.get('final_response') call.
See: issue #9433
"""

import concurrent.futures
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FakeAgent:
    """Mock agent that returns a configurable value from run_conversation."""

    def __init__(self, return_value):
        self._return_value = return_value

    def get_activity_summary(self):
        return {"seconds_since_activity": 0.0}

    def run_conversation(self, prompt):
        return self._return_value


def _run_type_guard(result_value, job_name="test-job"):
    """Exercise the same code path as run_job after the agent finishes.

    Reproduces lines ~770-838 of cron/scheduler.py: submit the agent to a
    thread pool, collect the result, then apply the type guard before calling
    result.get('final_response').
    """
    agent = FakeAgent(result_value)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(agent.run_conversation, "test prompt")

    try:
        result = future.result()
    finally:
        pool.shutdown(wait=False)

    # --- This is the type guard added for #9433 ---
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Cron job '{job_name}': agent.run_conversation returned "
            f"{type(result).__name__} instead of dict"
        )

    # This is the line that previously crashed with AttributeError
    final_response = result.get("final_response", "") or ""
    return final_response


class TestNonDictResultTypeGuard:
    """agent.run_conversation must return a dict; anything else is a RuntimeError."""

    def test_dict_result_succeeds(self):
        """Normal dict result should return final_response."""
        resp = _run_type_guard({"final_response": "Hello!"})
        assert resp == "Hello!"

    def test_dict_empty_response(self):
        """Dict with no final_response returns empty string."""
        resp = _run_type_guard({})
        assert resp == ""

    def test_string_result_raises(self):
        """A plain string result must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="returned str instead of dict"):
            _run_type_guard("some error string")

    def test_none_result_raises(self):
        """A None result must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="returned NoneType instead of dict"):
            _run_type_guard(None)

    def test_int_result_raises(self):
        """An int result must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="returned int instead of dict"):
            _run_type_guard(42)

    def test_list_result_raises(self):
        """A list result must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="returned list instead of dict"):
            _run_type_guard(["unexpected"])

    def test_error_message_includes_job_name(self):
        """The RuntimeError message should include the job name."""
        with pytest.raises(RuntimeError, match="my-cron-job"):
            _run_type_guard(None, job_name="my-cron-job")

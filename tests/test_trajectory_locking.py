"""Tests that concurrent save_trajectory() calls produce valid JSONL."""

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from agent.trajectory import save_trajectory


def test_concurrent_writes_produce_valid_jsonl():
    """Multiple threads writing simultaneously must not corrupt the file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jsonl")
        n_writers = 20

        def _write(i):
            save_trajectory(
                trajectory=[{"from": "human", "value": f"msg-{i}"}],
                model="test-model",
                completed=True,
                filename=path,
            )

        with ThreadPoolExecutor(max_workers=n_writers) as pool:
            list(pool.map(_write, range(n_writers)))

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == n_writers
        for line in lines:
            entry = json.loads(line)  # must not raise
            assert "conversations" in entry
            assert entry["completed"] is True


def test_single_write_produces_valid_jsonl():
    """Sanity check: a single write is valid JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.jsonl")
        save_trajectory(
            trajectory=[{"from": "human", "value": "hello"}],
            model="m",
            completed=False,
            filename=path,
        )
        with open(path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["model"] == "m"
        assert entry["completed"] is False

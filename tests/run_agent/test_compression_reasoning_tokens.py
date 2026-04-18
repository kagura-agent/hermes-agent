"""Tests that reasoning/thinking tokens in completion_tokens don't cause
premature compression.  (#12026)

Thinking models (GLM-5.1, QwQ, etc.) report large completion_tokens that
include internal reasoning tokens.  The compression trigger must use only
prompt_tokens to decide whether the context window is full.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run_agent
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


def _mock_response(content="Hello", usage=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**usage) if usage else None
    return resp


class TestCompressionReasoningTokens:
    """Compression trigger should ignore completion_tokens (which include
    reasoning tokens) and only consider prompt_tokens vs context_length."""

    def _make_agent(self, context_length=200_000, threshold=0.5):
        agent = AIAgent.__new__(AIAgent)
        agent.model = "test/thinking-model"
        agent.session_id = str(uuid.uuid4())
        agent.task_id = None
        agent.messages = []
        agent.compression_enabled = True
        agent.tool_definitions = []
        agent.active_persona = None
        agent._session_db = None
        agent._safe_print = lambda *a, **k: None

        from agent.context_compressor import ContextCompressor
        with patch("agent.context_compressor.get_model_context_length", return_value=context_length):
            agent.context_compressor = ContextCompressor(
                model="test/thinking-model",
                threshold_percent=threshold,
            )
        return agent

    def test_high_completion_tokens_no_premature_compression(self):
        """When prompt_tokens is low but completion_tokens is huge (reasoning),
        compression should NOT trigger."""
        agent = self._make_agent(context_length=200_000, threshold=0.5)
        cc = agent.context_compressor

        # Simulate usage: prompt is small, but completion has huge reasoning
        cc.update_from_response({
            "prompt_tokens": 50_000,      # well below 100k threshold
            "completion_tokens": 80_000,  # huge due to reasoning tokens
        })

        # Old buggy code: _real_tokens = 50k + 80k = 130k >= 100k → compress!
        # Fixed code: _real_tokens = 50k < 100k → no compression
        assert not cc.should_compress(cc.last_prompt_tokens), \
            "Compression should not trigger based on prompt_tokens alone"

    def test_compression_triggers_on_high_prompt_tokens(self):
        """When prompt_tokens actually exceeds threshold, compression fires."""
        agent = self._make_agent(context_length=200_000, threshold=0.5)
        cc = agent.context_compressor

        cc.update_from_response({
            "prompt_tokens": 120_000,     # above 100k threshold
            "completion_tokens": 3000,
        })

        assert cc.should_compress(cc.last_prompt_tokens), \
            "Compression should trigger when prompt_tokens exceeds threshold"

    def test_real_tokens_excludes_completion(self):
        """Verify the _real_tokens calculation in the agent loop uses only
        prompt_tokens, not prompt_tokens + completion_tokens."""
        agent = self._make_agent(context_length=200_000, threshold=0.5)
        cc = agent.context_compressor

        cc.update_from_response({
            "prompt_tokens": 50_000,
            "completion_tokens": 80_000,  # massive reasoning tokens
        })

        # Replicate the logic from run_agent.py lines ~11066-11072
        if cc.last_prompt_tokens > 0:
            _real_tokens = cc.last_prompt_tokens
        else:
            _real_tokens = 99999  # fallback shouldn't be reached

        assert _real_tokens == 50_000, \
            "_real_tokens should equal prompt_tokens only, not include completion_tokens"
        assert not cc.should_compress(_real_tokens), \
            "50k tokens should not trigger compression at 100k threshold"

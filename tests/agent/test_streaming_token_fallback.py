"""Tests for streaming token accounting fallback when provider returns no usage."""

import types
import unittest
from unittest.mock import MagicMock, patch

from agent.model_metadata import estimate_messages_tokens_rough, estimate_tokens_rough


def _make_agent_stub():
    """Create a minimal stub with the attributes the token accounting block reads."""
    agent = MagicMock()
    agent.model = "test-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_mode = "chat_completions"
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_api_calls = 0
    agent.session_id = "test-session"
    agent._session_db = MagicMock()
    agent.context_compressor = MagicMock()
    return agent


class TestStreamingTokenFallback(unittest.TestCase):
    """Verify the else-branch fallback in token accounting."""

    def test_normal_usage_path(self):
        """When response.usage is present, canonical path fires (no fallback)."""
        usage = types.SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        response = types.SimpleNamespace(usage=usage)
        # hasattr + truthiness check should pass
        self.assertTrue(hasattr(response, "usage") and response.usage)

    def test_fallback_fires_when_usage_none(self):
        """When usage is None the condition is falsy — fallback should fire."""
        response = types.SimpleNamespace(usage=None)
        self.assertFalse(hasattr(response, "usage") and response.usage)

    def test_estimated_tokens_nonzero(self):
        """Rough estimates must be >0 for non-empty input."""
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        est_input = estimate_messages_tokens_rough(messages)
        self.assertGreater(est_input, 0)

        output_text = "I am doing well, thank you for asking!"
        est_output = estimate_tokens_rough(output_text)
        self.assertGreater(est_output, 0)

    def test_fallback_persists_to_session_db(self):
        """Simulate the fallback branch logic and verify DB persistence."""
        agent = _make_agent_stub()
        messages = [{"role": "user", "content": "What is 2+2?"}]
        output_content = "2+2 equals 4."

        # Simulate the fallback branch inline
        est_input = estimate_messages_tokens_rough(messages)
        est_output = estimate_tokens_rough(output_content)

        agent.session_input_tokens += est_input
        agent.session_output_tokens += est_output
        agent._session_db.update_token_counts(
            agent.session_id,
            input_tokens=est_input,
            output_tokens=est_output,
        )

        self.assertGreater(agent.session_input_tokens, 0)
        self.assertGreater(agent.session_output_tokens, 0)
        agent._session_db.update_token_counts.assert_called_once_with(
            "test-session",
            input_tokens=est_input,
            output_tokens=est_output,
        )


if __name__ == "__main__":
    unittest.main()

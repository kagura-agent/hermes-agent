"""Tests that assistant messages with content=None preserve None instead of
normalizing to empty string ''.

Anthropic-compatible proxies reject empty string content with HTTP 400:
'text content blocks must be non-empty'.  Tool-call-only responses from the
model return content=None, which must stay None in the conversation history.

Fixes: #11906
"""

import types
import unittest


def _make_assistant_msg(content, tool_calls=None, reasoning_content=None):
    """Create a minimal mock assistant message object."""
    return types.SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        reasoning_content=reasoning_content,
    )


class TestAgentLoopNoneContent(unittest.TestCase):
    """agent_loop.py must not convert content=None to '' in message dicts."""

    def test_tool_call_message_preserves_none_content(self):
        """The assistant message dict built for tool-call turns must keep content=None."""
        from environments.agent_loop import HermesAgentLoop
        import inspect
        source = inspect.getsource(HermesAgentLoop.run)

        # Find lines with content or "" normalization, excluding the read-only
        # fallback parser check (which just tests string membership).
        lines = [
            line.strip() for line in source.splitlines()
            if 'assistant_msg.content or ""' in line
            and "<tool_call>" not in line
        ]
        self.assertEqual(
            len(lines), 0,
            f"Found message-dict normalization of None to '': {lines}",
        )

class TestBuildAssistantMessageNoneContent(unittest.TestCase):
    """run_agent.py _build_assistant_message must preserve content=None."""

    def _make_agent(self):
        from run_agent import AIAgent
        agent = AIAgent.__new__(AIAgent)
        agent.verbose_logging = False
        agent.reasoning_callback = None
        agent.stream_delta_callback = None
        agent._stream_callback = None
        return agent

    def test_none_content_stays_none(self):
        """_build_assistant_message should keep content=None, not convert to ''."""
        agent = self._make_agent()
        msg = _make_assistant_msg(content=None)
        result = agent._build_assistant_message(msg, finish_reason="stop")

        self.assertIsNone(
            result["content"],
            f"content should be None, got {result['content']!r}",
        )

    def test_string_content_preserved(self):
        """_build_assistant_message should preserve string content."""
        agent = self._make_agent()
        msg = _make_assistant_msg(content="Hello")
        result = agent._build_assistant_message(msg, finish_reason="stop")
        self.assertEqual(result["content"], "Hello")

    def test_empty_string_content_normalized_to_none(self):
        """_build_assistant_message should normalize empty string to None (#11906)."""
        agent = self._make_agent()
        msg = _make_assistant_msg(content="")
        result = agent._build_assistant_message(msg, finish_reason="stop")
        self.assertIsNone(result["content"])


if __name__ == "__main__":
    unittest.main()

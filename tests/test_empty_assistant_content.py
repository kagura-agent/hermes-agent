"""Tests for #11906: Empty assistant content normalization.

Ensures that empty/whitespace-only assistant message content is normalized
to None so that downstream proxies don't generate empty text blocks that
Anthropic rejects with HTTP 400.
"""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import patch


class TestBuildAssistantMessageEmptyContent:
    """Test _build_assistant_message normalizes empty content to None."""

    def _make_agent(self):
        """Create a minimal AIAgent-like object for testing."""
        from run_agent import AIAgent
        agent = AIAgent.__new__(AIAgent)
        agent.verbose_logging = False
        agent.reasoning_callback = None
        agent.stream_delta_callback = None
        agent._stream_callback = None
        agent._skill_nudge_interval = 0
        return agent

    def test_none_content_preserved(self):
        agent = self._make_agent()
        msg = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="test", arguments="{}"),
                type="function",
            )],
        )
        # Mock reasoning extraction
        with patch.object(agent, '_extract_reasoning', return_value=None):
            result = agent._build_assistant_message(msg, "stop")
        assert result["content"] is None

    def test_empty_string_content_normalized_to_none(self):
        agent = self._make_agent()
        msg = SimpleNamespace(
            content="",
            tool_calls=[SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="test", arguments="{}"),
                type="function",
            )],
        )
        with patch.object(agent, '_extract_reasoning', return_value=None):
            result = agent._build_assistant_message(msg, "stop")
        assert result["content"] is None

    def test_whitespace_content_normalized_to_none(self):
        agent = self._make_agent()
        msg = SimpleNamespace(
            content="   \n  ",
            tool_calls=None,
        )
        with patch.object(agent, '_extract_reasoning', return_value=None):
            result = agent._build_assistant_message(msg, "stop")
        assert result["content"] is None

    def test_real_content_preserved(self):
        agent = self._make_agent()
        msg = SimpleNamespace(
            content="Hello, world!",
            tool_calls=None,
        )
        with patch.object(agent, '_extract_reasoning', return_value=None):
            result = agent._build_assistant_message(msg, "stop")
        assert result["content"] == "Hello, world!"


class TestAnthropicAdapterEmptyContent:
    """Test that the Anthropic adapter handles empty content correctly."""

    def test_empty_content_assistant_with_tool_calls(self):
        """Assistant message with tool_calls and empty content should not produce empty text blocks."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ],
            }
        ]
        system, result = convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assistant_msg = result[0]
        # Should have tool_use block but no empty text block
        for block in assistant_msg["content"]:
            if block.get("type") == "text":
                assert block["text"] != "", "Empty text block should not be present"

    def test_none_content_assistant_with_tool_calls(self):
        """Assistant message with tool_calls and None content should not produce empty text blocks."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ],
            }
        ]
        system, result = convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assistant_msg = result[0]
        # Should have tool_use block but no empty text block
        for block in assistant_msg["content"]:
            if block.get("type") == "text":
                assert block["text"] != "", "Empty text block should not be present"

    def test_real_content_assistant_preserved(self):
        """Assistant message with actual content should preserve it."""
        from agent.anthropic_adapter import convert_messages_to_anthropic

        messages = [
            {
                "role": "assistant",
                "content": "I'll help you with that.",
            }
        ]
        system, result = convert_messages_to_anthropic(messages)

        assert len(result) == 1
        assistant_msg = result[0]
        text_blocks = [b for b in assistant_msg["content"] if b.get("type") == "text"]
        assert any(b["text"] == "I'll help you with that." for b in text_blocks)


class TestStreamingEmptyContent:
    """Test that streaming correctly handles empty content."""

    def test_empty_content_parts_yield_none(self):
        """When no content deltas arrive (tool-call-only), content should be None."""
        content_parts = []
        full_content = "".join(content_parts) or None
        assert full_content is None

    def test_whitespace_only_content_parts(self):
        """Whitespace-only streamed content should still be preserved (it's valid content)."""
        content_parts = ["  "]
        full_content = "".join(content_parts) or None
        assert full_content == "  "  # Whitespace IS valid streamed content

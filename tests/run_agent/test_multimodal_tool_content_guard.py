"""Tests for multimodal tool content provider guard.

Covers the fix for #27344: vision-capable models whose providers do not
accept list-type ``content`` in tool messages should receive a text
summary instead of the raw multimodal content parts.
"""

from __future__ import annotations

from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(provider: str = "openai", model: str = "gpt-4o",
                api_mode: str = "chat_completions") -> AIAgent:
    """Build a bare-bones AIAgent for pure-method tests."""
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.api_mode = api_mode
    agent._anthropic_image_fallback_cache = {}
    return agent


MULTIMODAL_RESULT = {
    "_multimodal": True,
    "text_summary": "Screenshot captured",
    "content": [
        {"type": "text", "text": "Screenshot captured"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ],
}


# ─── _provider_supports_multimodal_tool_content ──────────────────────────────


class TestProviderSupportsMultimodalToolContent:
    """Verify the allowlist correctly identifies supported providers."""

    def test_anthropic_messages_api_mode(self):
        agent = _make_agent(provider="anthropic", api_mode="anthropic_messages")
        assert agent._provider_supports_multimodal_tool_content() is True

    def test_codex_responses_api_mode(self):
        agent = _make_agent(provider="openai-codex", api_mode="codex_responses")
        assert agent._provider_supports_multimodal_tool_content() is True

    def test_gemini_native_api_mode(self):
        agent = _make_agent(provider="google", api_mode="gemini_native")
        assert agent._provider_supports_multimodal_tool_content() is True

    def test_openai_chat_completions(self):
        agent = _make_agent(provider="openai", api_mode="chat_completions")
        assert agent._provider_supports_multimodal_tool_content() is True

    def test_azure_chat_completions(self):
        agent = _make_agent(provider="azure", api_mode="chat_completions")
        assert agent._provider_supports_multimodal_tool_content() is True

    def test_xiaomi_chat_completions_not_supported(self):
        agent = _make_agent(provider="xiaomi", api_mode="chat_completions")
        assert agent._provider_supports_multimodal_tool_content() is False

    def test_deepseek_chat_completions_not_supported(self):
        agent = _make_agent(provider="deepseek", api_mode="chat_completions")
        assert agent._provider_supports_multimodal_tool_content() is False

    def test_openrouter_chat_completions_not_supported(self):
        agent = _make_agent(provider="openrouter", api_mode="chat_completions")
        assert agent._provider_supports_multimodal_tool_content() is False

    def test_custom_provider_not_supported(self):
        agent = _make_agent(provider="my-custom-llm", api_mode="chat_completions")
        assert agent._provider_supports_multimodal_tool_content() is False


# ─── _tool_result_content_for_active_model ───────────────────────────────────


class TestToolResultContentGuard:
    """Verify the end-to-end tool result guard for #27344."""

    def test_vision_supported_provider_returns_content(self):
        """OpenAI + vision = multimodal content passed through."""
        agent = _make_agent(provider="openai", api_mode="chat_completions")
        with patch.object(agent, "_model_supports_vision", return_value=True):
            result = agent._tool_result_content_for_active_model(
                "computer_use", MULTIMODAL_RESULT,
            )
        # Should return the content list directly
        assert isinstance(result, list)
        types = [p.get("type") for p in result]
        assert "image_url" in types

    def test_vision_unsupported_provider_returns_summary(self):
        """Xiaomi MiMo has vision but provider rejects list tool content."""
        agent = _make_agent(provider="xiaomi", model="mimo-v2.5",
                            api_mode="chat_completions")
        with patch.object(agent, "_model_supports_vision", return_value=True):
            result = agent._tool_result_content_for_active_model(
                "computer_use", MULTIMODAL_RESULT,
            )
        # Should fall back to text summary, not the list
        assert isinstance(result, str)
        assert "Screenshot captured" in result

    def test_non_vision_model_returns_error_for_computer_use(self):
        """Non-vision model gets a JSON error for computer_use."""
        agent = _make_agent(provider="deepseek", model="deepseek-chat",
                            api_mode="chat_completions")
        with patch.object(agent, "_model_supports_vision", return_value=False):
            result = agent._tool_result_content_for_active_model(
                "computer_use", MULTIMODAL_RESULT,
            )
        assert isinstance(result, str)
        assert "does not support image input" in result

    def test_non_vision_model_returns_summary_for_other_tools(self):
        """Non-vision model gets text summary for non-computer_use tools."""
        agent = _make_agent(provider="deepseek", model="deepseek-chat",
                            api_mode="chat_completions")
        with patch.object(agent, "_model_supports_vision", return_value=False):
            result = agent._tool_result_content_for_active_model(
                "vision_analyze", MULTIMODAL_RESULT,
            )
        assert isinstance(result, str)
        assert "Screenshot captured" in result

    def test_non_multimodal_result_passes_through(self):
        """Plain string results are not affected by the guard."""
        agent = _make_agent(provider="xiaomi", api_mode="chat_completions")
        result = agent._tool_result_content_for_active_model(
            "terminal", "command output here",
        )
        assert result == "command output here"

    def test_anthropic_api_mode_bypasses_provider_check(self):
        """anthropic_messages api_mode always supports multimodal tool content."""
        agent = _make_agent(provider="some-proxy", api_mode="anthropic_messages")
        with patch.object(agent, "_model_supports_vision", return_value=True):
            result = agent._tool_result_content_for_active_model(
                "computer_use", MULTIMODAL_RESULT,
            )
        assert isinstance(result, list)

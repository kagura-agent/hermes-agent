from agent.smart_model_routing import choose_cheap_model_route, _DEFAULT_COMPLEX_KEYWORDS


_BASE_CONFIG = {
    "enabled": True,
    "cheap_model": {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
}


def test_returns_none_when_disabled():
    cfg = {**_BASE_CONFIG, "enabled": False}
    assert choose_cheap_model_route("what time is it in tokyo?", cfg) is None


def test_routes_short_simple_prompt():
    result = choose_cheap_model_route("what time is it in tokyo?", _BASE_CONFIG)
    assert result is not None
    assert result["provider"] == "openrouter"
    assert result["model"] == "google/gemini-2.5-flash"
    assert result["routing_reason"] == "simple_turn"


def test_skips_long_prompt():
    prompt = "please summarize this carefully " * 20
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_code_like_prompt():
    prompt = "debug this traceback: ```python\nraise ValueError('bad')\n```"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_skips_tool_heavy_prompt_keywords():
    prompt = "implement a patch for this docker error"
    assert choose_cheap_model_route(prompt, _BASE_CONFIG) is None


def test_resolve_turn_route_falls_back_to_primary_when_route_runtime_cannot_be_resolved(monkeypatch):
    from agent.smart_model_routing import resolve_turn_route

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad route")),
    )
    result = resolve_turn_route(
        "what time is it in tokyo?",
        _BASE_CONFIG,
        {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-primary",
        },
    )
    assert result["model"] == "anthropic/claude-sonnet-4"
    assert result["runtime"]["provider"] == "openrouter"
    assert result["label"] is None


# --- complex_keywords_extra / complex_keywords_override tests ---


def test_extra_keywords_block_routing():
    """complex_keywords_extra adds new keywords that block cheap routing."""
    cfg = {**_BASE_CONFIG, "complex_keywords_extra": ["banana"]}
    # "banana" is not a default keyword, so without extra it routes cheap
    assert choose_cheap_model_route("banana", {**_BASE_CONFIG}) is not None
    # With extra it should be blocked
    assert choose_cheap_model_route("banana", cfg) is None


def test_override_replaces_defaults():
    """complex_keywords_override fully replaces the default keyword set."""
    cfg = {**_BASE_CONFIG, "complex_keywords_override": ["banana"]}
    # "debug" is a default keyword but not in the override list
    assert choose_cheap_model_route("debug", cfg) is not None
    # "banana" is in the override list
    assert choose_cheap_model_route("banana", cfg) is None


def test_override_plus_extra():
    """extra merges on top of override when both are set."""
    cfg = {**_BASE_CONFIG, "complex_keywords_override": ["banana"], "complex_keywords_extra": ["mango"]}
    assert choose_cheap_model_route("banana", cfg) is None
    assert choose_cheap_model_route("mango", cfg) is None
    # default keyword not in either list
    assert choose_cheap_model_route("debug", cfg) is not None


def test_defaults_unchanged_without_config():
    """Without override/extra, the default keyword set is used."""
    assert choose_cheap_model_route("debug", _BASE_CONFIG) is None
    assert choose_cheap_model_route("kubernetes", _BASE_CONFIG) is None

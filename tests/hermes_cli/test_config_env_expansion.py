"""Tests for ${ENV_VAR} substitution in config.yaml values."""

import os
import pytest
import yaml
from hermes_cli.config import (
    _expand_env_vars,
    _collect_env_placeholders,
    _restore_env_placeholders,
    load_config,
    save_config,
    read_raw_config,
)
from unittest.mock import patch as mock_patch


class TestExpandEnvVars:
    def test_simple_substitution(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("MY_KEY", "secret123")
            assert _expand_env_vars("${MY_KEY}") == "secret123"

    def test_missing_var_kept_verbatim(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("UNDEFINED_VAR_XYZ", raising=False)
            assert _expand_env_vars("${UNDEFINED_VAR_XYZ}") == "${UNDEFINED_VAR_XYZ}"

    def test_no_placeholder_unchanged(self):
        assert _expand_env_vars("plain-value") == "plain-value"

    def test_dict_recursive(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("TOKEN", "tok-abc")
            result = _expand_env_vars({"key": "${TOKEN}", "other": "literal"})
            assert result == {"key": "tok-abc", "other": "literal"}

    def test_nested_dict(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("API_KEY", "sk-xyz")
            result = _expand_env_vars({"model": {"api_key": "${API_KEY}"}})
            assert result["model"]["api_key"] == "sk-xyz"

    def test_list_items(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("VAL", "hello")
            result = _expand_env_vars(["${VAL}", "literal", 42])
            assert result == ["hello", "literal", 42]

    def test_non_string_values_untouched(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(3.14) == 3.14
        assert _expand_env_vars(True) is True
        assert _expand_env_vars(None) is None

    def test_multiple_placeholders_in_one_string(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("HOST", "localhost")
            mp.setenv("PORT", "5432")
            assert _expand_env_vars("${HOST}:${PORT}") == "localhost:5432"

    def test_dict_keys_not_expanded(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("KEY", "value")
            result = _expand_env_vars({"${KEY}": "no-expand-key"})
            assert "${KEY}" in result


class TestLoadConfigExpansion:
    def test_load_config_expands_env_vars(self, tmp_path, monkeypatch):
        config_yaml = (
            "model:\n"
            "  api_key: ${GOOGLE_API_KEY}\n"
            "platforms:\n"
            "  telegram:\n"
            "    token: ${TELEGRAM_BOT_TOKEN}\n"
            "plain: no-substitution\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("GOOGLE_API_KEY", "gsk-test-key")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1234567:ABC-token")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["api_key"] == "gsk-test-key"
        assert config["platforms"]["telegram"]["token"] == "1234567:ABC-token"
        assert config["plain"] == "no-substitution"

    def test_load_config_unresolved_kept_verbatim(self, tmp_path, monkeypatch):
        config_yaml = "model:\n  api_key: ${NOT_SET_XYZ_123}\n"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.delenv("NOT_SET_XYZ_123", raising=False)
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)

        config = load_config()

        assert config["model"]["api_key"] == "${NOT_SET_XYZ_123}"


class TestLoadCliConfigExpansion:
    """Verify that load_cli_config() also expands ${VAR} references."""

    def test_cli_config_expands_auxiliary_api_key(self, tmp_path, monkeypatch):
        config_yaml = (
            "auxiliary:\n"
            "  vision:\n"
            "    api_key: ${TEST_VISION_KEY_XYZ}\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("TEST_VISION_KEY_XYZ", "vis-key-123")
        # Patch the hermes home so load_cli_config finds our test config
        monkeypatch.setattr("cli._hermes_home", tmp_path)

        from cli import load_cli_config
        config = load_cli_config()

        assert config["auxiliary"]["vision"]["api_key"] == "vis-key-123"

    def test_cli_config_unresolved_kept_verbatim(self, tmp_path, monkeypatch):
        config_yaml = (
            "auxiliary:\n"
            "  vision:\n"
            "    api_key: ${UNSET_CLI_VAR_ABC}\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.delenv("UNSET_CLI_VAR_ABC", raising=False)
        monkeypatch.setattr("cli._hermes_home", tmp_path)

        from cli import load_cli_config
        config = load_cli_config()

        assert config["auxiliary"]["vision"]["api_key"] == "${UNSET_CLI_VAR_ABC}"


class TestCollectEnvPlaceholders:
    def test_flat_dict(self):
        raw = {"api_key": "${SECRET}", "name": "literal"}
        result = _collect_env_placeholders(raw)
        assert result == {"api_key": "${SECRET}"}

    def test_nested_dict(self):
        raw = {"providers": {"my-llm": {"api_key": "${MY_KEY}", "api": "http://x"}}}
        result = _collect_env_placeholders(raw)
        assert result == {"providers.my-llm.api_key": "${MY_KEY}"}

    def test_list_items(self):
        raw = {"keys": ["${A}", "literal"]}
        result = _collect_env_placeholders(raw)
        assert result == {"keys[0]": "${A}"}

    def test_no_placeholders(self):
        assert _collect_env_placeholders({"x": "plain", "n": 42}) == {}

    def test_multiple_placeholders_in_one_string(self):
        raw = {"url": "https://${HOST}:${PORT}/api"}
        result = _collect_env_placeholders(raw)
        assert result == {"url": "https://${HOST}:${PORT}/api"}


class TestRestoreEnvPlaceholders:
    def test_restores_nested(self):
        placeholders = {"providers.my-llm.api_key": "${MY_KEY}"}
        expanded = {"providers": {"my-llm": {"api_key": "actual-secret", "api": "http://x"}}}
        result = _restore_env_placeholders(expanded, placeholders)
        assert result["providers"]["my-llm"]["api_key"] == "${MY_KEY}"
        assert result["providers"]["my-llm"]["api"] == "http://x"

    def test_no_placeholders_passthrough(self):
        data = {"a": "b", "c": 1}
        assert _restore_env_placeholders(data, {}) == data

    def test_list_restore(self):
        placeholders = {"keys[0]": "${A}"}
        expanded = {"keys": ["expanded-a", "literal"]}
        result = _restore_env_placeholders(expanded, placeholders)
        assert result["keys"] == ["${A}", "literal"]


class TestSaveConfigPreservesPlaceholders:
    """Round-trip: config with ${…} placeholders survives save_config()."""

    def test_migration_round_trip(self, tmp_path, monkeypatch):
        """api_key placeholder must survive load_config → save_config cycle."""
        config_yaml = (
            "providers:\n"
            "  my-llm:\n"
            "    api: http://localhost:8080\n"
            "    api_key: ${MY_SECRET_KEY}\n"
            "model:\n"
            "  default: gpt-4\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("MY_SECRET_KEY", "sk-real-secret-value")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)
        monkeypatch.setattr("hermes_cli.config.ensure_hermes_home", lambda: None)
        monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)

        # Simulate what migrate_config does: load → modify → save
        config = load_config()
        assert config["providers"]["my-llm"]["api_key"] == "sk-real-secret-value"

        save_config(config)

        # Re-read the raw file — placeholder must be preserved
        raw = yaml.safe_load(config_file.read_text())
        assert raw["providers"]["my-llm"]["api_key"] == "${MY_SECRET_KEY}"
        assert "sk-real-secret-value" not in config_file.read_text()

    def test_non_placeholder_values_written_normally(self, tmp_path, monkeypatch):
        """Literal api_key values (no ${…}) should be written as-is."""
        config_yaml = (
            "providers:\n"
            "  my-llm:\n"
            "    api: http://localhost:8080\n"
            "    api_key: sk-literal-key\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)
        monkeypatch.setattr("hermes_cli.config.ensure_hermes_home", lambda: None)
        monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)

        config = load_config()
        save_config(config)

        raw = yaml.safe_load(config_file.read_text())
        assert raw["providers"]["my-llm"]["api_key"] == "sk-literal-key"

    def test_multiple_placeholders_preserved(self, tmp_path, monkeypatch):
        """Multiple placeholders across different keys all survive."""
        config_yaml = (
            "providers:\n"
            "  openai:\n"
            "    api_key: ${OPENAI_KEY}\n"
            "  anthropic:\n"
            "    api_key: ${ANTHROPIC_KEY}\n"
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_yaml)

        monkeypatch.setenv("OPENAI_KEY", "sk-openai-123")
        monkeypatch.setenv("ANTHROPIC_KEY", "sk-ant-456")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_file)
        monkeypatch.setattr("hermes_cli.config.ensure_hermes_home", lambda: None)
        monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)

        config = load_config()
        save_config(config)

        raw = yaml.safe_load(config_file.read_text())
        assert raw["providers"]["openai"]["api_key"] == "${OPENAI_KEY}"
        assert raw["providers"]["anthropic"]["api_key"] == "${ANTHROPIC_KEY}"
        file_text = config_file.read_text()
        assert "sk-openai-123" not in file_text
        assert "sk-ant-456" not in file_text

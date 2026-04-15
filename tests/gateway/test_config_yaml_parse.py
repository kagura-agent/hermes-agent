"""Test that the gateway main() function parses YAML config files correctly."""

import textwrap
from unittest.mock import patch

from gateway.config import GatewayConfig


class TestMainConfigYamlParse:
    def test_main_parses_yaml_config(self, tmp_path):
        """main() should parse a YAML config file via --config flag."""
        config_file = tmp_path / "gateway.yaml"
        config_file.write_text(textwrap.dedent("""\
            platforms:
              telegram:
                enabled: true
                token: "test-token-123"
            session_reset:
              mode: idle
              idle_minutes: 60
        """))

        captured = {}

        async def fake_start_gateway(config):
            captured["config"] = config
            return True

        with (
            patch("sys.argv", ["run", "--config", str(config_file)]),
            patch("gateway.run.start_gateway", fake_start_gateway),
        ):
            from gateway.run import main
            main()

        config = captured["config"]
        assert isinstance(config, GatewayConfig)
        assert config.platforms  # at least one platform parsed

    def test_main_no_config_flag_passes_none(self):
        """main() should pass None to start_gateway when --config is omitted."""
        captured = {}

        async def fake_start_gateway(config):
            captured["config"] = config
            return True

        with (
            patch("sys.argv", ["run"]),
            patch("gateway.run.start_gateway", fake_start_gateway),
        ):
            from gateway.run import main
            main()

        assert captured["config"] is None

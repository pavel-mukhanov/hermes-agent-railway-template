import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("ADMIN_PASSWORD", "test-password")

import server


class ModelConfigSyncTests(unittest.TestCase):
    def test_sync_model_config_creates_model_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            env_vars = {
                "HERMES_MODEL": "anthropic/claude-opus-4.6",
                "OPENROUTER_API_KEY": "sk-test",
            }

            with patch.object(server, "CONFIG_FILE_PATH", config_path):
                server.sync_model_config_file(env_vars, "test")

            self.assertEqual(
                config_path.read_text(),
                "model:\n"
                "  default: 'anthropic/claude-opus-4.6'\n"
                "  provider: 'openrouter'\n",
            )

    def test_sync_model_config_replaces_only_top_level_model_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "display:\n"
                "  tool_progress: 'off'\n"
                "model:\n"
                "  default: 'old/model'\n"
                "  provider: 'openrouter'\n"
                "  context_length: 131072\n"
                "delegation:\n"
                "  model: 'delegate/model'\n"
            )
            env_vars = {
                "HERMES_MODEL": "google/gemini-3-flash-preview",
                "OPENROUTER_API_KEY": "sk-test",
            }

            with patch.object(server, "CONFIG_FILE_PATH", config_path):
                server.sync_model_config_file(env_vars, "test")

            self.assertEqual(
                config_path.read_text(),
                "display:\n"
                "  tool_progress: 'off'\n"
                "model:\n"
                "  default: 'google/gemini-3-flash-preview'\n"
                "  provider: 'openrouter'\n"
                "  context_length: 131072\n"
                "delegation:\n"
                "  model: 'delegate/model'\n",
            )

    def test_sync_model_config_falls_back_to_existing_config_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "model:\n"
                "  default: 'existing/model'\n"
                "  provider: 'openrouter'\n"
            )

            with patch.object(server, "CONFIG_FILE_PATH", config_path):
                server.sync_model_config_file({"OPENROUTER_API_KEY": "sk-test"}, "test")

            self.assertEqual(
                config_path.read_text(),
                "model:\n"
                "  default: 'existing/model'\n"
                "  provider: 'openrouter'\n",
            )


if __name__ == "__main__":
    unittest.main()

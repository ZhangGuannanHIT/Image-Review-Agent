# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.llm import DeepSeekConfig, LLMClient
from backend.mock import MockOCRClient, MockReviewEngine, RuleBasedMockChatModel
from backend.service import ReviewService


ROOT = Path(__file__).resolve().parents[1]


class DeepSeekConfigTests(unittest.TestCase):
    def test_project_config_uses_deepseek_v4_flash(self):
        config = DeepSeekConfig.from_yaml(str(ROOT / "config" / "llm.yaml"))
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertEqual(config.base_url, "https://api.deepseek.com")

    def test_environment_key_overrides_blank_file_value(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
            config = DeepSeekConfig.from_yaml(str(ROOT / "config" / "llm.yaml"))
        self.assertEqual(config.api_key, "test-key")

    def test_missing_key_has_clear_error(self):
        client = LLMClient(config=DeepSeekConfig(api_key=""))
        with self.assertRaisesRegex(RuntimeError, "LLM_API_KEY"):
            client.build_model()

    def test_rag_disabled_removes_tool(self):
        with tempfile.TemporaryDirectory() as log_dir:
            service = ReviewService(
                RuleBasedMockChatModel(),
                MockReviewEngine(),
                MockOCRClient(),
                log_dir=log_dir,
                rag_enabled=False,
            )
            self.assertNotIn("search_knowledge", {tool.name for tool in service.tools})
            self.assertIn("analyze_image", {tool.name for tool in service.tools})


if __name__ == "__main__":
    unittest.main()

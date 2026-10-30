# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import tools as tools_mod
from backend.llm import DeepSeekConfig, LLMClient
from backend.mock import MockOCRClient, MockReviewEngine, RuleBasedMockChatModel
from backend.service import ReviewService


ROOT = Path(__file__).resolve().parents[1]
LLM_EXAMPLE = ROOT / "config" / "llm.example.yaml"


class DeepSeekConfigTests(unittest.TestCase):
    def test_project_config_uses_deepseek_v4_flash(self):
        config = DeepSeekConfig.from_yaml(str(LLM_EXAMPLE))
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertEqual(config.base_url, "https://api.deepseek.com")

    def test_environment_key_overrides_blank_file_value(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
            config = DeepSeekConfig.from_yaml(str(LLM_EXAMPLE))
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

    def test_rag_enabled_registers_and_caches_retrieval(self):
        calls = []

        def fake_retriever(query, k=5):
            calls.append((query, k))
            return [{"id": "REG-TEST", "title": "测试规则"}]

        tools = tools_mod.build_tools(
            MockReviewEngine(),
            MockOCRClient(),
            fake_retriever,
            enable_rag=True,
        )
        analyze = next(tool for tool in tools if tool.name == "analyze_image")
        search = next(tool for tool in tools if tool.name == "search_knowledge")
        token = tools_mod.set_current_image(b"test-image")
        try:
            analyze.invoke({})
            first = search.invoke({"query": "直接暴力审核规则"})
            second = search.invoke({"query": "直接暴力审核规则"})
            cache_keys = list(tools_mod.get_request_context().cache)
        finally:
            tools_mod.reset_current_image(token)
        self.assertEqual(calls, [("直接暴力审核规则", 5)])
        self.assertEqual(first, second)
        self.assertIn("REG-TEST", first)
        self.assertFalse(any("直接暴力审核规则" in key for key in cache_keys))

    def test_rag_requires_clip_evidence(self):
        tools = tools_mod.build_tools(MockReviewEngine(), MockOCRClient(), enable_rag=True)
        search = next(tool for tool in tools if tool.name == "search_knowledge")
        token = tools_mod.set_current_image(b"test-image")
        try:
            with self.assertRaisesRegex(RuntimeError, "必须先成功调用 analyze_image"):
                search.invoke({"query": "直接暴力审核规则"})
        finally:
            tools_mod.reset_current_image(token)

    def test_service_uses_fixed_workflow_without_llm_tool_selection(self):
        with tempfile.TemporaryDirectory() as log_dir:
            service = ReviewService(
                RuleBasedMockChatModel(),
                MockReviewEngine(),
                MockOCRClient(),
                log_dir=log_dir,
            )
        self.assertTrue(hasattr(service, "workflow"))
        self.assertFalse(hasattr(service, "agent"))


if __name__ == "__main__":
    unittest.main()

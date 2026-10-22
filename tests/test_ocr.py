# -*- coding: utf-8 -*-

import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from backend.ocr import AliyunOCRConfig, GLMOCRConfig, OCRClient


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 16), "white").save(output, format="PNG")
    return output.getvalue()


class GLMOCRTests(unittest.TestCase):
    def test_example_config_is_aliyun_market_ocr(self):
        config = AliyunOCRConfig.from_yaml("config/ocr.example.yaml")
        self.assertEqual(config.provider, "aliyun_market_ocr")
        self.assertEqual(config.endpoint, "https://gjbsb.market.alicloudapi.com/ocrservice/advanced")

    def test_legacy_glm_config_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ocr.yaml"
            path.write_text(
                "glm_ocr:\n  provider: glm_ocr\n  model: glm-ocr\n  api_key: secret\n",
                encoding="utf-8",
            )
            client = OCRClient.from_config(path)
            self.assertEqual(client.config.provider, "glm_ocr")

    def test_not_configured_is_not_no_text(self):
        result = OCRClient(GLMOCRConfig(api_key="")).recognize(png_bytes())
        self.assertFalse(result["success"])
        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "not_configured")

    @patch("backend.ocr.requests.post")
    def test_official_request_and_layout_response(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "model": "GLM-OCR",
            "request_id": "req-1",
            "md_results": "测试文字",
            "layout_details": [[
                {"label": "text", "content": "测试文字", "bbox_2d": [0.1, 0.2, 0.8, 0.4]},
                {"label": "image", "content": "ignored", "bbox_2d": [0, 0, 1, 1]},
            ]],
        }
        post.return_value = response
        client = OCRClient(GLMOCRConfig(api_key="secret"))
        result = client.recognize(png_bytes())

        self.assertTrue(result["success"])
        self.assertEqual(result["texts"], ["测试文字"])
        self.assertEqual(result["boxes"], [[0.1, 0.2, 0.8, 0.4]])
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["json"]["model"], "glm-ocr")
        file_value = kwargs["json"]["file"]
        self.assertTrue(file_value.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(file_value.split(",", 1)[1]), png_bytes())
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")

    @patch("backend.ocr.requests.post")
    def test_markdown_fallback(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"md_results": "只有 Markdown", "layout_details": []}
        post.return_value = response
        result = OCRClient(GLMOCRConfig(api_key="secret")).recognize(png_bytes())
        self.assertEqual(result["texts"], ["只有 Markdown"])
        self.assertEqual(result["boxes"], [])

    @patch("backend.ocr.requests.post")
    def test_aliyun_appcode_request_and_response(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "sid": "sid-1",
            "width": 100,
            "height": 50,
            "content": "测试文字",
            "prism_rowsInfo": [{"rowId": 0, "word": "测试文字"}],
            "prism_wordsInfo": [{
                "word": "测试文字",
                "prob": 88,
                "pos": [
                    {"x": 10, "y": 5},
                    {"x": 80, "y": 5},
                    {"x": 80, "y": 25},
                    {"x": 10, "y": 25},
                ],
            }],
        }
        post.return_value = response
        client = OCRClient(AliyunOCRConfig(appcode="secret"))
        result = client.recognize(png_bytes())
        self.assertTrue(result["success"])
        self.assertEqual(result["texts"], ["测试文字"])
        self.assertEqual(result["boxes"], [[0.1, 0.1, 0.8, 0.5]])
        self.assertAlmostEqual(result["mean_confidence"], 0.88)
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Authorization"], "APPCODE secret")
        self.assertTrue(kwargs["json"]["prob"])
        self.assertNotIn("url", kwargs["json"])


if __name__ == "__main__":
    unittest.main()

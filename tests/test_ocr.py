# -*- coding: utf-8 -*-

import base64
import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from backend.ocr import GLMOCRConfig, OCRClient


ROOT = Path(__file__).resolve().parents[1]
OCR_EXAMPLE = ROOT / "config" / "ocr.example.yaml"


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 16), "white").save(output, format="PNG")
    return output.getvalue()


class GLMOCRTests(unittest.TestCase):
    def test_config_is_glm_ocr(self):
        config = GLMOCRConfig.from_yaml(str(OCR_EXAMPLE))
        self.assertEqual(config.provider, "glm_ocr")
        self.assertEqual(config.model, "glm-ocr")

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


if __name__ == "__main__":
    unittest.main()

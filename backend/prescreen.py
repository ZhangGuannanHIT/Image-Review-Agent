# -*- coding: utf-8 -*-
"""已废弃的旧文字预筛兼容模块；新架构由 Agent 按需调用 OCR。"""

from __future__ import annotations

from typing import Optional

# 两句模板（可写进 config 便于标定）。含"是否带文字"对比。
HAS_TEXT_SENTENCES = ["图片画面中包含文字、字幕、标语或小广告字样"]
NO_TEXT_SENTENCES = ["图片画面中不包含任何文字"]


def has_text(image, engine=None, threshold: float = 0.5) -> bool:
    """判断图片是否含文字。

    依赖外部传入的 ReviewEngine（用于取图文特征）。engine 未提供时返回默认 False
    （保守：不触发 OCR，避免误调 API）。
    """
    return False

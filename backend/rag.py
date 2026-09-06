# -*- coding: utf-8 -*-
"""RAG 检索接口（本期仅占位）。

判例库 + 向量检索本期不做，接口签名固定、返回空。
后续实现时替换内部逻辑，签名与返回结构保持稳定。
"""

from __future__ import annotations

from typing import Any


def retrieve(query_text: str, k: int = 5) -> list[dict]:
    """检索相似判例。

    本期返回 []（占位）。后续实现（判例库 + 向量检索）后返回：
      [{"id": "#A42", "profile": "...", "desc": "...", "verdict": "违规", "basis": "..."}]
    """
    # TODO: 实现判例库 + 向量检索（faiss/chroma + FG-CLIP2 文本编码）
    return []

# -*- coding: utf-8 -*-
"""轻量本地 RAG：字符 n-gram TF-IDF + 余弦相似度。"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "knowledge.json"
_ITEM_TYPES = {"regulation", "verified_case", "teaching_case", "ocr_boundary"}
_LOCATOR_STATUSES = {
    "exact_url_verified", "site_search_only", "placeholder_unverified", "attachment_only",
}
_REQUIRED_FIELDS = {"id", "type", "title", "tags", "content", "verdict", "basis", "source"}
_MIN_SCORE = 0.30  # 当前 55 条校准查询中，不降低召回的最高整档阈值


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(char for char in normalized if char.isalnum())


def _ngrams(text: str) -> Counter[str]:
    normalized = _normalize(text)
    grams: Counter[str] = Counter()
    for size in (2, 3, 4):
        grams.update(normalized[index:index + size] for index in range(len(normalized) - size + 1))
    return grams


def _validate_item(item: Any, index: int, seen: set[str]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"知识条目 {index} 必须是对象")
    missing = sorted(_REQUIRED_FIELDS - item.keys())
    if missing:
        raise ValueError(f"知识条目 {index} 缺少字段: {missing}")
    item_id = item["id"]
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError(f"知识条目 {index} 的 id 无效")
    if item_id in seen:
        raise ValueError(f"知识库存在重复 id: {item_id}")
    seen.add(item_id)
    if item["type"] not in _ITEM_TYPES:
        raise ValueError(f"知识条目 {item_id} 的 type 无效: {item['type']}")
    if not isinstance(item["tags"], list) or not all(isinstance(tag, str) for tag in item["tags"]):
        raise ValueError(f"知识条目 {item_id} 的 tags 必须是字符串数组")
    for field in ("title", "content", "verdict", "basis"):
        if not isinstance(item[field], str) or not item[field].strip():
            raise ValueError(f"知识条目 {item_id} 的 {field} 无效")
    source = item["source"]
    if not isinstance(source, dict):
        raise ValueError(f"知识条目 {item_id} 的 source 必须是对象")
    for field in ("title", "publisher", "locator", "locator_status", "note"):
        if not isinstance(source.get(field), str):
            raise ValueError(f"知识条目 {item_id} 的 source.{field} 无效")
    status = source["locator_status"]
    if status not in _LOCATOR_STATUSES:
        raise ValueError(f"知识条目 {item_id} 的 locator_status 无效: {status}")
    if status == "exact_url_verified":
        locator = source["locator"]
        parsed = urlparse(locator)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or "xxx" in locator.casefold():
            raise ValueError(f"知识条目 {item_id} 的精确来源地址无效")
    return item


def load_items(path: str | Path = KNOWLEDGE_PATH) -> list[dict[str, Any]]:
    """加载并校验规则、判例和 OCR 边界知识。"""
    knowledge_path = Path(path)
    with knowledge_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("知识库根节点必须是数组")
    seen: set[str] = set()
    items = [_validate_item(item, index, seen) for index, item in enumerate(payload)]
    if not items:
        raise ValueError("知识库不能为空")
    return items


def _item_chunks(item: dict[str, Any]) -> list[str]:
    """建立正向检索文本块；反例只随结果返回，不参与相似度评分。"""
    return [item["title"], *item["tags"], item["content"]]


class LocalKnowledgeRetriever:
    """面向小型中文知识库的内存检索器，不依赖外部向量服务。"""

    def __init__(self, items: list[dict[str, Any]], min_score: float = _MIN_SCORE):
        if not items:
            raise ValueError("知识库不能为空")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score 必须在 0 到 1 之间")
        self.items = [dict(item) for item in items]
        self.min_score = min_score

        chunk_counters: list[Counter[str]] = []
        self.chunk_owners: list[int] = []
        for item_index, item in enumerate(self.items):
            for chunk in _item_chunks(item):
                counter = _ngrams(chunk)
                if counter:
                    chunk_counters.append(counter)
                    self.chunk_owners.append(item_index)

        document_frequency: Counter[str] = Counter()
        for counter in chunk_counters:
            document_frequency.update(counter.keys())
        total = len(chunk_counters)
        self.idf = {
            gram: math.log((total + 1) / (frequency + 1)) + 1.0
            for gram, frequency in document_frequency.items()
        }
        self.vectors = [self._vectorize(counter) for counter in chunk_counters]
        self.norms = [self._norm(vector) for vector in self.vectors]

    @classmethod
    def from_json(
        cls, path: str | Path = KNOWLEDGE_PATH, min_score: float = _MIN_SCORE,
    ) -> "LocalKnowledgeRetriever":
        return cls(load_items(path), min_score=min_score)

    def _vectorize(self, counter: Counter[str]) -> dict[str, float]:
        return {
            gram: (1.0 + math.log(count)) * self.idf[gram]
            for gram, count in counter.items()
            if gram in self.idf and count > 0
        }

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    def score(self, query_text: str) -> list[tuple[float, str]]:
        """返回未应用阈值的条目最高相关度，供离线校准使用。"""
        if len(_normalize(query_text)) < 2:
            return []
        query_vector = self._vectorize(_ngrams(query_text))
        query_norm = self._norm(query_vector)
        if not query_vector or query_norm == 0:
            return []

        item_scores = [0.0] * len(self.items)
        for owner, vector, norm in zip(self.chunk_owners, self.vectors, self.norms):
            if norm == 0:
                continue
            shared = query_vector.keys() & vector.keys()
            score = sum(query_vector[gram] * vector[gram] for gram in shared) / (query_norm * norm)
            item_scores[owner] = max(item_scores[owner], score)
        return sorted(
            ((score, self.items[index]["id"]) for index, score in enumerate(item_scores) if score > 0),
            key=lambda result: (-result[0], result[1]),
        )

    def retrieve(self, query_text: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        ranked = [result for result in self.score(query_text) if result[0] >= self.min_score]

        results: list[dict[str, Any]] = []
        items_by_id = {item["id"]: item for item in self.items}
        for score, item_id in ranked[:k]:
            item = items_by_id[item_id]
            result = dict(item)
            result["source"] = dict(item["source"])
            result["tags"] = list(item["tags"])
            result["profile"] = item["type"]
            result["desc"] = item["content"]
            result["score"] = round(score, 6)
            results.append(result)
        return results


@lru_cache(maxsize=1)
def _default_retriever() -> LocalKnowledgeRetriever:
    return LocalKnowledgeRetriever.from_json()


def warmup() -> int:
    """启动时加载并校验默认知识库，返回可用条目数。"""
    return len(_default_retriever().items)


def retrieve(query_text: str, k: int = 5) -> list[dict]:
    """检索相关审核规则、脱敏判例和 OCR 语义边界。"""
    return _default_retriever().retrieve(query_text, k=k)

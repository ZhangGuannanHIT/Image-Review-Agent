"""Structured OCR readability and RAG evidence."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Callable


HIGH_RISK_RULE_IDS = {"R001", "R002", "R003", "R004", "R005", "R006", "R007", "R015"}
SAFE_CONTEXT_RULE_IDS = {"R009", "R010", "R011", "R013"}
IMAGE_DESCRIPTION_MARKERS = (
    "the image shows",
    "the image depicts",
    "this image shows",
    "in the image",
    "the picture shows",
    "画面中",
    "图中",
    "这张图",
)
SAFE_FOOD_TERMS = (
    "sauce",
    "soup",
    "stew",
    "tomato",
    "ketchup",
    "food",
    "bowl",
    "plate",
    "meal",
    "酱汁",
    "番茄酱",
    "番茄",
    "汤",
    "食物",
    "餐盘",
    "碗",
)
RED_LIQUID_TERMS = (
    "red sauce",
    "red liquid",
    "red paint",
    "tomato",
    "ketchup",
    "红色酱汁",
    "番茄酱",
    "红色液体",
    "红色颜料",
)
VISUAL_HARM_TERMS = (
    "blood",
    "wound",
    "gore",
    "injury",
    "corpse",
    "dead body",
    "murder",
    "attack",
    "weapon",
    "knife",
    "gun",
    "血迹",
    "伤口",
    "尸",
    "攻击",
    "武器",
    "刀",
    "枪",
)


def _normalized_text(texts: list[Any]) -> str:
    text = " ".join(str(item) for item in texts if str(item).strip())
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.strip()


def _alnum_count(text: str) -> int:
    return len(re.findall(r"[0-9A-Za-z\u4e00-\u9fff]", text))


def classify_ocr_content(text: str) -> dict[str, Any]:
    """Distinguish literal-looking OCR output from GLM's image captions."""
    normalized = text.casefold()
    descriptive = bool(
        any(marker in normalized for marker in IMAGE_DESCRIPTION_MARKERS)
    )
    safe_food = bool(
        any(term in normalized for term in SAFE_FOOD_TERMS)
        and not any(term in normalized for term in VISUAL_HARM_TERMS)
    )
    safe_red_liquid = bool(
        safe_food
        and any(term in normalized for term in RED_LIQUID_TERMS)
    )
    return {
        "descriptive_caption": descriptive,
        "safe_food_scene": safe_food,
        "safe_red_liquid": safe_red_liquid,
    }


def assess_text_evidence(
    ocr: dict[str, Any] | None,
    retrieve: Callable[..., list[dict]] | None = None,
) -> dict[str, Any]:
    """Return auditable text evidence without retaining raw OCR text."""
    if not isinstance(ocr, dict):
        return {
            "status": "not_available",
            "has_text": False,
            "char_count": 0,
            "text_sha256": "",
            "mean_confidence": None,
            "high_risk_rule_ids": [],
            "safe_context_rule_ids": [],
            "knowledge_hits": [],
            "content_profile": classify_ocr_content(""),
        }
    if not ocr.get("success"):
        return {
            "status": "failed",
            "has_text": False,
            "char_count": 0,
            "text_sha256": "",
            "mean_confidence": ocr.get("mean_confidence"),
            "high_risk_rule_ids": [],
            "safe_context_rule_ids": [],
            "knowledge_hits": [],
            "error": ocr.get("error"),
            "content_profile": classify_ocr_content(""),
        }

    texts = ocr.get("texts", []) or []
    text = _normalized_text(texts)
    char_count = _alnum_count(text)
    confidence = ocr.get("mean_confidence")
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None
    if char_count == 0:
        status = "empty"
    elif char_count < 5 or (
        confidence_value is not None and confidence_value < 0.65
    ):
        status = "partial"
    else:
        status = "readable"

    hits: list[dict[str, Any]] = []
    if status in {"partial", "readable"} and char_count >= 2 and retrieve is not None:
        try:
            hits = retrieve(text, k=5) or []
        except Exception:
            hits = []
    compact_hits = [
        {
            "id": str(item.get("id", "")),
            "type": str(item.get("type", "")),
            "score": item.get("score"),
            "verdict": str(item.get("verdict", ""))[:160],
        }
        for item in hits
        if isinstance(item, dict) and float(item.get("score", 0)) >= 0.30
    ]
    high_risk = [
        item["id"] for item in compact_hits if item["id"] in HIGH_RISK_RULE_IDS
    ]
    safe_context = [
        item["id"] for item in compact_hits if item["id"] in SAFE_CONTEXT_RULE_IDS
    ]
    return {
        "status": status,
        "has_text": char_count > 0,
        "char_count": char_count,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "mean_confidence": confidence_value,
        "high_risk_rule_ids": high_risk,
        "safe_context_rule_ids": safe_context,
        "knowledge_hits": compact_hits,
        "content_profile": classify_ocr_content(text),
    }


def collect_knowledge_hits(cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect and deduplicate cached knowledge-retrieval entries."""
    by_id: dict[str, dict[str, Any]] = {}
    for key, value in cache.items():
        if not key.startswith("search_knowledge:") or not isinstance(value, list):
            continue
        query_digest = key.split(":", 1)[1]
        for item in value:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            if not item_id:
                continue
            hit = {
                "id": item_id,
                "type": str(item.get("type", "")),
                "score": item.get("score"),
                "verdict": str(item.get("verdict", ""))[:160],
                "source_status": (
                    item.get("source", {}).get("locator_status", "")
                    if isinstance(item.get("source"), dict) else ""
                ),
                "query_digest": query_digest,
            }
            previous = by_id.get(item_id)
            if previous is None or float(hit["score"] or 0) > float(previous["score"] or 0):
                by_id[item_id] = hit
    return sorted(
        by_id.values(),
        key=lambda item: float(item.get("score") or 0),
        reverse=True,
    )


def build_decision_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Build the bounded audit bundle used by policy and explanation."""
    return {
        "model_heads": {
            "risk": evidence.get("risk", {}),
            "type": evidence.get("type", {}),
            "injury": evidence.get("injury", {}),
            "quality": evidence.get("quality", {}),
        },
        "blood_binary": evidence.get("blood_binary"),
        "anime_style": evidence.get("anime_style"),
        "text_detection": evidence.get(
            "text_detection", {"status": "not_available", "has_text": False}
        ),
        "lowlevel": evidence.get("lowlevel", {"status": "not_available"}),
        "ocr": evidence.get("text_evidence", {"status": "not_available"}),
        "scene_concepts": evidence.get(
            "scene_concepts", {"status": "not_called", "action_effect": "none"}
        ),
        "search_knowledge": evidence.get("search_knowledge", []),
    }

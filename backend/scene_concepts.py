# -*- coding: utf-8 -*-
"""Open-vocabulary scene concept scoring on frozen FG-CLIP2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_SCENE_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "scene_concepts.json"


def load_scene_concepts(path: str | Path = DEFAULT_SCENE_PATH) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        concepts = json.load(file)
    if not isinstance(concepts, list) or not concepts:
        raise ValueError("scene_concepts.json 必须是非空数组")
    seen = set()
    for concept in concepts:
        if concept.get("key") in seen:
            raise ValueError(f"场景概念 key 重复: {concept.get('key')}")
        seen.add(concept.get("key"))
        if not concept.get("prompts"):
            raise ValueError(f"场景概念 {concept.get('key')} 缺少 prompts")
        if concept.get("kind") not in {"risk", "safe_context", "confounder", "domain"}:
            raise ValueError(f"场景概念 {concept.get('key')} kind 无效")
    return concepts


def build_scene_evidence(
    result: dict,
    *,
    risk_margin_threshold: float,
    safe_context_threshold: float,
    confounder_threshold: float = 0.006,
    anime_score_threshold: float = 0.015,
    domain_margin_threshold: float = 0.01,
    top_k: int = 3,
    retrieve=None,
) -> dict:
    concepts = result["concepts"]
    risks = sorted(
        (
            {"key": key, **value}
            for key, value in concepts.items() if value["kind"] == "risk"
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]
    safe_contexts = sorted(
        (
            {"key": key, **value}
            for key, value in concepts.items() if value["kind"] == "safe_context"
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]
    confounders = sorted(
        (
            {"key": key, **value}
            for key, value in concepts.items() if value["kind"] == "confounder"
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]
    domains = sorted(
        (
            {"key": key, **value}
            for key, value in concepts.items() if value["kind"] == "domain"
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    risk_max = risks[0]["score"] if risks else -1.0
    competing_max = max(
        [item["score"] for item in safe_contexts + confounders] or [-1.0]
    )
    confounder_max = max(
        [item["score"] for item in confounders] or [-1.0]
    )
    risk_margin = risk_max - competing_max
    risk_matched = risk_margin >= risk_margin_threshold
    safe_matched = bool(
        safe_contexts and safe_contexts[0]["score"] >= safe_context_threshold
    )
    confounder_matched = bool(
        confounders and confounder_max >= confounder_threshold
    )
    domain_scores = {item["key"]: item["score"] for item in domains}
    anime_score = float(domain_scores.get("anime_illustration", -1.0))
    real_score = float(domain_scores.get("real_photo", -1.0))
    anime_margin = anime_score - real_score
    anime_matched = bool(
        anime_score >= anime_score_threshold
        and anime_margin >= domain_margin_threshold
    )
    domain = {
        "effective": "anime_illustration" if anime_matched else "unknown",
        "anime_matched": anime_matched,
        "anime_score": round(anime_score, 6),
        "real_score": round(real_score, 6),
        "anime_margin": round(anime_margin, 6),
        "anime_score_threshold": anime_score_threshold,
        "domain_margin_threshold": domain_margin_threshold,
        "candidates": domains,
    }
    conflict = bool(risk_matched and safe_matched)
    queries = []
    if risk_matched:
        queries.extend(item["rag_query"] for item in risks[:2] if item.get("rag_query"))
    if safe_matched:
        queries.extend(
            item["rag_query"] for item in safe_contexts[:2] if item.get("rag_query")
        )
    queries = list(dict.fromkeys(queries))
    allowed_rule_ids = set()
    for item in risks[:2] + safe_contexts[:2]:
        allowed_rule_ids.update(item.get("rag_ids", []))
    hits = []
    if retrieve is not None:
        for query in queries[:3]:
            try:
                for item in retrieve(query, k=3) or []:
                    if not isinstance(item, dict):
                        continue
                    hits.append({
                        "id": str(item.get("id", "")),
                        "type": str(item.get("type", "")),
                        "score": item.get("score"),
                        "verdict": str(item.get("verdict", ""))[:160],
                    })
            except Exception:
                continue
    unique_hits = []
    seen = set()
    for item in hits:
        if (
            item["id"]
            and item["id"] in allowed_rule_ids
            and item["id"] not in seen
        ):
            seen.add(item["id"])
            unique_hits.append(item)
    return {
        "status": "matched" if risk_matched or safe_matched else "uncertain",
        "risk_scenes": risks,
        "safe_contexts": safe_contexts,
        "confounders": confounders,
        "domains": domains,
        "domain": domain,
        "risk_margin": risk_margin,
        "risk_matched": risk_matched,
        "safe_context_matched": safe_matched,
        "confounder_matched": confounder_matched,
        "context_conflict": conflict,
        "rag_queries": queries,
        "knowledge_hits": unique_hits[:6],
        "action_effect": "effective_risk_context",
    }


class SceneConceptScorer:
    def __init__(self, engine, path: str | Path = DEFAULT_SCENE_PATH):
        from transformers import AutoTokenizer

        self.engine = engine
        self.concepts = load_scene_concepts(path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            engine.model_path, trust_remote_code=True, local_files_only=True
        )
        self.prompt_owner = []
        prompts = []
        for concept in self.concepts:
            for prompt in concept["prompts"]:
                prompts.append(prompt)
                self.prompt_owner.append(concept["key"])
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(engine.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        position_ids = torch.arange(
            input_ids.shape[-1], device=engine.device, dtype=torch.long
        ).unsqueeze(0)
        with torch.inference_mode():
            features = engine.model.base.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                walk_type="short",
            )
        self.text_features = torch.nn.functional.normalize(
            features.float(), dim=-1
        )

    def _base_image_features(self, image: Any) -> torch.Tensor:
        inputs = self.engine.prepare_image(image)
        vision = self.engine.model.vision
        context = vision.disable_adapter() if hasattr(vision, "disable_adapter") else None
        with torch.inference_mode():
            if context is None:
                features = self.engine.model.forward_vision(
                    inputs["pixel_values"],
                    inputs["pixel_attention_mask"],
                    inputs["spatial_shapes"],
                )
            else:
                with context:
                    features = self.engine.model.forward_vision(
                        inputs["pixel_values"],
                        inputs["pixel_attention_mask"],
                        inputs["spatial_shapes"],
                    )
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def score(self, image: Any) -> dict:
        image_features = self._base_image_features(image)
        similarities = torch.matmul(self.text_features, image_features[0])
        grouped = {}
        offset = 0
        for concept in self.concepts:
            count = len(concept["prompts"])
            values = similarities[offset:offset + count]
            offset += count
            grouped[concept["key"]] = {
                "title": concept["title"],
                "kind": concept["kind"],
                "score": float(values.mean().item()),
                "max": float(values.max().item()),
                "rag_query": concept.get("rag_query", ""),
                "rag_ids": concept.get("rag_ids", []),
            }
        ranked = sorted(grouped.items(), key=lambda item: item[1]["score"], reverse=True)
        return {
            "concepts": grouped,
            "top": [
                {"key": key, **value}
                for key, value in ranked[:8]
            ],
            "raw_prompt_count": int(self.text_features.shape[0]),
        }

"""Fixed conditional evidence workflow for a single image review."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from . import tools as tools_mod
from .text_evidence import assess_text_evidence
from .threat_rule import threat_rule


class FixedReviewWorkflow:
    """Run tools in a deterministic order without an LLM controller."""

    version = "2026-10-26-fixed-evidence-v1"

    def __init__(
        self,
        tools: list[Any],
        *,
        ocr_enabled: bool = True,
        scene_concepts_enabled: bool = False,
        rag_enabled: bool = True,
        retriever: Any = None,
        text_semantic_reviewer: Any = None,
    ):
        self.tool_map = tools_mod.tools_by_name(tools)
        self.ocr_enabled = bool(ocr_enabled)
        self.scene_concepts_enabled = bool(scene_concepts_enabled)
        self.rag_enabled = bool(rag_enabled)
        self.retriever = retriever
        self.text_semantic_reviewer = text_semantic_reviewer

    @staticmethod
    def _step_result(value: Any) -> Any:
        return value

    def run(self) -> dict[str, Any]:
        context = tools_mod.get_request_context()
        steps: list[dict[str, Any]] = []
        tool_results: dict[str, Any] = {}
        degraded = False

        def call(name: str, *, required: bool = False) -> Any:
            nonlocal degraded
            tool = self.tool_map.get(name)
            if tool is None:
                if required:
                    raise RuntimeError(f"required tool missing: {name}")
                return None
            started = time.monotonic()
            try:
                value = tool.invoke({})
                steps.append({
                    "type": "tool_call",
                    "name": name,
                    "ok": True,
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                })
                if name == "detect_text":
                    cache_value = context.cache.get("text_detection", value)
                else:
                    cache_value = context.cache.get(name, value)
                tool_results[name] = self._step_result(cache_value)
                return cache_value
            except Exception as exc:  # noqa: BLE001
                degraded = True
                steps.append({
                    "type": "tool_error",
                    "name": name,
                    "ok": False,
                    "error": type(exc).__name__,
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                })
                if required:
                    raise
                return None

        # CLIP is the only mandatory visual evidence source.
        call("analyze_image", required=True)
        evidence = context.cache.get("analyze_image", {})
        risk_label = evidence.get("risk", {}).get("label", "")

        if "review_blood" in self.tool_map:
            call("review_blood")

        detection = None
        if "detect_text" in self.tool_map:
            call("detect_text")
            detection = context.cache.get("text_detection")
            if detection and detection.get("status") == "failed":
                degraded = True

        text_gate = (
            detection is None
            or detection.get("status") in {"detected", "failed", "unavailable"}
            or detection.get("has_text")
        )
        ocr_called = False
        if not self.ocr_enabled:
            steps.append({
                "type": "tool_skip",
                "name": "ocr_image",
                "reason": "ocr_disabled",
            })
        elif text_gate:
            call("ocr_image")
            ocr_called = True
        else:
            steps.append({
                "type": "tool_skip",
                "name": "ocr_image",
                "reason": "text_detector_empty",
            })

        if ocr_called:
            started = time.monotonic()
            assessment = assess_text_evidence(
                context.cache.get("ocr_image"),
                self.retriever if self.rag_enabled else None,
            )
            ocr_result = context.cache.get("ocr_image") or {}
            full_text = "\n".join(
                str(item) for item in (ocr_result.get("texts") or [])
                if str(item).strip()
            )
            semantic = (
                self.text_semantic_reviewer.review(
                    full_text, assessment.get("knowledge_hits", [])
                )
                if self.text_semantic_reviewer and full_text
                else {"status": "not_available", "verdict": "uncertain"}
            )
            rule = threat_rule(full_text) if full_text else None
            quote = str(semantic.get("quote", ""))
            grounded = bool(
                semantic.get("status") == "ok"
                and semantic.get("verdict") == "violation"
                and len(quote.strip()) >= 3
                and quote in full_text
            )
            assessment["semantic_review"] = {
                "status": semantic.get("status"),
                "verdict": semantic.get("verdict"),
                "prompt_version": semantic.get("prompt_version"),
                "quote_sha256": (
                    hashlib.sha256(quote.encode("utf-8")).hexdigest()
                    if quote else ""
                ),
                "quote_char_count": len(quote),
                "target_present": bool(semantic.get("target")),
                "harm_action_present": bool(semantic.get("harm_action")),
                "stance_present": bool(semantic.get("stance")),
                "error": semantic.get("error"),
            }
            assessment["grounded_violation"] = bool(grounded or rule)
            assessment["deterministic_rule_id"] = (
                rule.get("rule_id") if rule else ""
            )
            if rule:
                assessment["rule_evidence_sha256"] = hashlib.sha256(
                    str(rule.get("evidence", "")).encode("utf-8")
                ).hexdigest()
            if full_text and semantic.get("status") == "invalid":
                degraded = True
            context.cache["text_evidence"] = assessment
            steps.append({
                "type": "tool_call",
                "name": "retrieve_text_rules",
                "ok": True,
                "status": assessment.get("status"),
                "rule_count": len(
                    assessment.get("high_risk_rule_ids", [])
                    + assessment.get("safe_context_rule_ids", [])
                ),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            })
            steps.append({
                "type": "text_semantic_review",
                "status": semantic.get("status"),
                "verdict": semantic.get("verdict"),
                "grounded_violation": assessment["grounded_violation"],
                "deterministic_rule": assessment["deterministic_rule_id"],
            })

        # Violation scene prompts are useful only for non-L0 visual candidates.
        if self.scene_concepts_enabled and risk_label != "L0_安全":
            call("match_scene_concepts")
        elif self.scene_concepts_enabled:
            steps.append({
                "type": "tool_skip",
                "name": "match_scene_concepts",
                "reason": "risk_l0",
            })

        return {
            "workflow_version": self.version,
            "clip_called": "analyze_image" in tool_results,
            "clip_succeeded": "analyze_image" in context.cache,
            "finalized": True,
            "degraded": degraded,
            "steps": steps,
            "tool_results": tool_results,
        }

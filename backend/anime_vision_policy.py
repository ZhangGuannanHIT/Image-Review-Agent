"""Trigger and reconcile anime-only structured visual review."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.labels import RISK_CLASSES


def anime_visual_gate(
    decision: dict[str, Any],
    evidence: dict[str, Any],
    *,
    min_style_score: float = 0.90,
) -> dict[str, Any]:
    style = evidence.get("anime_style") or {}
    scene_domain = (
        (evidence.get("scene_concepts") or {}).get("domain") or {}
    )
    text = evidence.get("text_evidence") or {}
    semantic = text.get("semantic_review") or {}
    if isinstance(style, dict) and style:
        score = float(style.get("score") or 0)
        anime_route = score >= min_style_score
        route_source = "style_router"
    else:
        score = float(scene_domain.get("anime_score") or 0)
        anime_route = bool(scene_domain.get("anime_matched"))
        route_source = "scene_prompt"
    reasons = []
    if not anime_route:
        reasons.append("style_score_below_threshold")
    if decision.get("action") not in {"limit", "review", "block"}:
        reasons.append("action_not_reviewable")
    if text.get("grounded_violation"):
        reasons.append("grounded_text_violation")
    if (
        text.get("high_risk_rule_ids")
        and semantic.get("verdict") != "safe"
        and not text.get("grounded_violation")
    ):
        reasons.append("text_grounding_unresolved")
    return {
        "eligible": not reasons,
        "reason": ",".join(reasons) or "eligible",
        "style_score": score,
        "min_style_score": min_style_score,
        "route_source": route_source,
        "anime_route": anime_route,
        "action": decision.get("action"),
        "risk": decision.get("risk_raw"),
    }


def reconcile_anime_visual(
    decision: dict[str, Any],
    evidence: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(decision)
    result["anime_visual_review"] = review
    if review.get("status") != "ok":
        result["anime_visual_effect"] = "unchanged_review_failed"
        return result
    text = evidence.get("text_evidence") or {}
    if text.get("grounded_violation"):
        result["anime_visual_effect"] = "unchanged_grounded_text"
        return result

    injury = review.get("injury")
    attack = review.get("attack")
    threat = review.get("threat")
    weapon_visible = review.get("weapon_visible")
    context = review.get("context")
    appearance = review.get("appearance")
    game_context = context in {"game_ui", "game_gameplay"}
    fiction_score = max(
        [
            float(item.get("score", -1))
            for item in (
                (evidence.get("scene_concepts") or {}).get("safe_contexts", [])
            )
            if item.get("key") == "fiction_context"
        ]
        or [-1.0]
    )
    fictional_action = bool(
        fiction_score >= 0.04
        and attack == "present"
        and injury == "absent"
        and threat != "present"
        and appearance == "cartoon"
    )
    if (
        (game_context or fictional_action)
        and injury == "absent"
        and threat != "present"
    ):
        result.update(
            action="allow",
            effective_risk=RISK_CLASSES[0],
            level=RISK_CLASSES[0],
            uncertain=False,
            conflict_type="anime_game_context",
            reason="明确游戏或虚构语境，未见伤害或对他人的明确威胁，撤回候选限制",
        )
        result["anime_visual_effect"] = "withdraw_game_context"
        result["anime_visual_resolution"] = "allow"
        return result
    if attack == "present" or threat == "present" or injury in {
        "moderate", "severe"
    }:
        result["anime_visual_effect"] = "visual_risk_supported"
        result["anime_visual_resolution"] = "keep"
        return result
    # No visible injury, attack or maintained threat. Dark/horror style remains
    # a mild age-guidance boundary; ordinary/cartoon display can be withdrawn.
    if (
        attack == "uncertain"
        or threat == "uncertain"
        or weapon_visible == "uncertain"
        or injury == "uncertain"
        or appearance == "horror"
        or injury == "mild"
        or (
            weapon_visible == "present"
            and context not in {"game_ui", "game_gameplay"}
        )
        or result.get("risk_raw") == RISK_CLASSES[3]
    ):
        result.update(
            action="limit",
            effective_risk=RISK_CLASSES[1],
            level=RISK_CLASSES[1],
            uncertain=False,
            conflict_type="anime_visual_mild_boundary",
            reason="动漫视觉复核未确认明确伤害、攻击或威胁，按 L1 限流且不转人工",
        )
        result["anime_visual_effect"] = "withdraw_to_l1"
        result["anime_visual_resolution"] = "limit"
        return result

    result.update(
        action="allow",
        effective_risk=RISK_CLASSES[0],
        level=RISK_CLASSES[0],
        uncertain=False,
        conflict_type="anime_visual_withdrawn",
        reason="动漫视觉复核未见伤害、攻击或明确武器指向，撤回候选限流",
    )
    result["anime_visual_effect"] = "withdraw_to_allow"
    result["anime_visual_resolution"] = "allow"
    return result

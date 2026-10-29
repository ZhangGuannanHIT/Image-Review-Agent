"""Trigger and reconcile anime-only structured visual review."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.labels import RISK_CLASSES, TYPE_CLASSES


def _sync_recommendation(
    result: dict[str, Any],
    action: str,
    reason: str = "",
) -> None:
    result.update(
        recommended_action=action,
        recommended_action_label={
            "allow": "建议放行",
            "limit": "建议限流",
            "review": "建议人工判断",
            "block": "建议下架",
        }[action],
        review_hint=action == "review",
        review_reason=reason if action == "review" else "",
    )


def anime_visual_gate(
    decision: dict[str, Any],
    evidence: dict[str, Any],
    *,
    min_style_score: float = 0.90,
    allow_safety_review: bool = False,
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
        anime_route_source = "style_router"
    else:
        score = float(scene_domain.get("anime_score") or 0)
        anime_route = bool(scene_domain.get("anime_matched"))
        anime_route_source = "scene_prompt"
    scene = evidence.get("scene_concepts") or {}
    game_ui_score = float(scene.get("game_ui_score") or -1.0)
    game_ui_matched = bool(scene.get("game_ui_matched"))
    type_labels = set((evidence.get("type") or {}).get("labels") or [])
    weapon_signal = TYPE_CLASSES[3] in type_labels
    game_ui_route = bool(game_ui_matched and weapon_signal)
    route_matched = bool(anime_route or game_ui_route)
    # Keep the established anime route when both signals fire. The game-UI
    # prompt is a fallback for real-looking game screens, not a replacement
    # for an already matched virtual-style route.
    route_source = anime_route_source if anime_route else "game_ui_prompt"
    reasons = []
    safety_signals: list[str] = []
    if not route_matched:
        reasons.append("visual_route_not_matched")
    action = decision.get("action")
    if action in {"limit", "review", "block"}:
        action_reviewable = True
    elif action == "allow" and allow_safety_review:
        risk = decision.get("risk_raw")
        quality_labels = set(
            (evidence.get("quality") or {}).get("labels") or []
        )
        if risk in {RISK_CLASSES[2], RISK_CLASSES[3]}:
            safety_signals.append("elevated_risk_on_allow")
        if quality_labels & {
            "严重模糊",
            "故意遮挡（马赛克/黑条/贴纸）",
            "文字扭曲形变",
        }:
            safety_signals.append("quality_uncertainty_on_allow")
        if decision.get("binary_state") in {"positive", "uncertain"}:
            safety_signals.append("binary_uncertainty_on_allow")
        action_reviewable = bool(safety_signals)
        if not action_reviewable:
            reasons.append("allow_without_safety_signal")
    else:
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
        "game_ui_route": game_ui_route,
        "game_ui_score": game_ui_score,
        "game_ui_matched": game_ui_matched,
        "weapon_signal": weapon_signal,
        "route_matched": route_matched,
        "action": decision.get("action"),
        "risk": decision.get("risk_raw"),
        "allow_safety_review": bool(allow_safety_review),
        "safety_signals": safety_signals,
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
    trigger_route = review.get("trigger_route")
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
    if trigger_route == "game_ui_prompt" and not game_context:
        result["anime_visual_effect"] = "unchanged_non_game_context"
        result["anime_visual_resolution"] = "keep"
        return result
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
        _sync_recommendation(result, "allow")
        result["anime_visual_effect"] = "withdraw_game_context"
        result["anime_visual_resolution"] = "allow"
        return result
    if attack == "present" or threat == "present" or injury in {
        "moderate", "severe"
    }:
        if result.get("action") == "allow":
            escalated_risk = (
                RISK_CLASSES[2]
                if injury in {"moderate", "severe"}
                else RISK_CLASSES[1]
            )
            result.update(
                action="limit",
                effective_risk=escalated_risk,
                level=escalated_risk,
                uncertain=False,
                conflict_type="anime_visual_safety_escalation",
                reason="虚拟内容初步放行后，视觉复核发现明确伤害、攻击或威胁，改为限流",
            )
            _sync_recommendation(result, "limit")
            result["anime_visual_effect"] = "visual_risk_escalated_from_allow"
            result["anime_visual_resolution"] = "limit"
            return result
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
        _sync_recommendation(result, "limit")
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
    _sync_recommendation(result, "allow")
    result["anime_visual_effect"] = "withdraw_to_allow"
    result["anime_visual_resolution"] = "allow"
    return result

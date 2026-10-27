"""Narrow conditional-threat rule; never a keyword-only classifier."""

from __future__ import annotations

import re


def threat_rule(text: str):
    # Invitation/condition followed by a first-person violent response.
    normalized = " ".join(str(text).lower().split())
    exclusions = (
        r"\b(news|reported|reporting|novel|fiction|fictional|movie|film|game|"
        r"photograph\w*|camera|portrait\w*|landscape\w*|sunset|basketball|"
        r"hoops|pool|billiards|skeet|clay pigeons|paper targets?|"
        r"shooting range|condemns?|opposes?|do not|don.t|never|not going to)\b|"
        r"新闻|通报|小说|虚构|电影|游戏|摄影|相机|靶场|纸靶|反对|不要|不会"
    )
    if re.search(exclusions, normalized):
        return None
    cue = re.search(
        r"\b(?:please|if you|when you|you|whoever|anyone who)\s+"
        r"(?:loot|steal|rob|trespass)\b",
        normalized,
    )
    harm = re.search(
        r"\b(?:i|we)\s+(?:will\s+|shall\s+|am going to\s+|are going to\s+|"
        r"love to\s+)?(?:shoot|kill|stab|beat)\b",
        normalized,
    )
    if (
        cue
        and harm
        and cue.start() < harm.start()
        and harm.end() - cue.start() <= 180
    ):
        return {
            "rule_id": "conditional_first_person_violence_v1",
            "evidence": str(text),
            "reason": "条件或邀请行为与第一人称伤害响应形成同一段威胁。",
        }
    return None

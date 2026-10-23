"""Gated image tools and evidence-scoped decision reconciliation."""
import base64
import io
import json
from copy import deepcopy

from PIL import Image, ImageOps
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator
from typing import Literal

from .evidence_policy import verified_text
from .review_prompts import VISUAL_PROMPT, TEXT_PROMPT


class VisualEvidence(BaseModel):
    injury: Literal['absent', 'mild', 'moderate', 'severe', 'uncertain']
    attack: Literal['present', 'absent', 'uncertain']
    threat: Literal['present', 'absent', 'uncertain']
    appearance: Literal['ordinary', 'cartoon', 'horror', 'uncertain']
    threat_basis: Literal['pointed_at_person', 'raised_to_strike', 'forced_restraint', 'none', 'uncertain']
    threat_target: str
    weapon_pose: str
    reason: str = Field(min_length=1, max_length=800)

    @model_validator(mode='after')
    def require_relation(self):
        if self.threat == 'present' and (self.threat_basis in {'none', 'uncertain'}
                or self.threat_target.strip() in {'', '无', '未知', 'none'}):
            raise ValueError('A threat requires a visible relation and target')
        return self


class TextEvidence(BaseModel):
    text: str = Field(max_length=8000)
    verdict: Literal['safe', 'violation', 'uncertain']
    quote: str = Field(max_length=8000)
    reason: str = Field(min_length=1, max_length=800)

    @model_validator(mode='after')
    def require_quote(self):
        if self.verdict == 'violation' and (len(self.quote.strip()) < 3 or self.quote not in self.text):
            raise ValueError('A textual violation requires an exact source quote')
        if '[?]' in self.text and self.verdict == 'safe':
            self.verdict = 'uncertain'
        return self


class EvidenceReviewer:
    def __init__(self, model):
        self.model = model.bind(response_format={'type': 'json_object'}, max_tokens=600,
                                extra_body={'thinking': {'type': 'disabled'}})

    def _review(self, image, prompt, schema, size):
        with Image.open(io.BytesIO(image)) as source:
            source = ImageOps.exif_transpose(source).convert('RGB')
            source.thumbnail((size, size))
            buffer = io.BytesIO()
            source.save(buffer, format='JPEG', quality=95)
        response = self.model.invoke([SystemMessage(content=prompt), HumanMessage(content=[
            {'type': 'text', 'text': '核验可见证据。'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' +
                base64.b64encode(buffer.getvalue()).decode(), 'detail': 'high'}}])])
        return {'status': 'ok', **schema.model_validate(json.loads(response.content)).model_dump()}

    def review_visual(self, image):
        return self._review(image, VISUAL_PROMPT, VisualEvidence, 1280)

    def review_text(self, image):
        return self._review(image, TEXT_PROMPT, TextEvidence, 1600)


def visual_gate(decision):
    return decision['action'] != 'allow' and not (decision.get('text_violation') or decision.get('rule_id'))


def visual_verdict(review):
    if not review or review.get('status') != 'ok':
        return 'uncertain'
    if (review.get('injury') in {'mild', 'moderate', 'severe'}
            or review.get('attack') == 'present' or review.get('threat') == 'present'):
        return 'supported'
    if all(review.get(k) == 'absent' for k in ('injury', 'attack', 'threat')):
        return 'not_supported'
    return 'uncertain'


def reconcile(decision, visual=None, text_review=None, *, visual_called=False,
              text_called=False, text_guard=False, text_hint=False):
    """Only withdraw visual evidence after independent text evidence is resolved."""
    result = deepcopy(decision)
    result['policy_version'] = '2026-10-23-evidence-review-v2'
    result['visual_evidence_review'] = visual
    result['text_evidence_review'] = text_review
    hints = []
    protected = bool(decision.get('text_violation') or decision.get('rule_id'))
    text_ok = bool(text_review and text_review.get('status') == 'ok')
    text_hit = text_ok and verified_text(text_review, text_review.get('text', ''))
    text_safe = text_ok and text_review.get('verdict') == 'safe'
    if text_hit:
        protected = True
        result.update(text_violation=True, uncertain=False, category='文字威胁',
                      reason='专项文字核验确认伤害威胁，按独立文字证据处置')
        if result['action'] == 'allow':
            result['action'] = 'limit'
    if visual_called:
        verdict = visual_verdict(visual)
        if verdict == 'not_supported' and not protected:
            if not text_guard or text_safe:
                result.update(action='allow', uncertain=False, category='未确认违规',
                              reason='专项核验未发现伤害、攻击或明确武器威胁',
                              visual_evidence_retracted=True)
            else:
                hints.append('画面证据已排除，但关键文字未核实，建议人工核对。')
        elif verdict == 'supported' and not protected:
            # A single review cannot promote limit to block.
            if result['action'] == 'allow':
                result['action'] = 'limit'
            basis = []
            if visual.get('injury') in {'mild', 'moderate', 'severe'}:
                basis.append('可辨认的伤害表现')
            if visual.get('attack') == 'present':
                basis.append('攻击行为')
            if visual.get('threat') == 'present':
                basis.append('明确武器威胁')
            result.update(reason='专项核验支持' + '、'.join(basis) + '，保留现有处置', uncertain=False)
        elif verdict == 'uncertain':
            hints.append('关键视觉证据无法核实，建议人工复核。')
    if (text_called and not text_safe and not text_hit and not protected
            and not any('文字' in x for x in hints)):
        hints.append('关键文字无法可靠辨认，建议人工核对原文。')
    elif text_hint and not text_safe and not text_hit and not protected and not text_called:
        hints.append('关键文字读数不一致，建议人工核对原文。')
    result.update(review_hint=bool(hints), review_reason=''.join(hints))
    return result


def render_evidence(decision):
    action = {'allow': '放行', 'limit': '限流', 'block': '下架'}[decision['action']]
    conclusion = '未确认违规' if decision['action'] == 'allow' else '存在违规依据'
    return f"审核结果：{conclusion}；处置：{action}。依据：{decision['reason'].rstrip('。')}。{decision.get('review_reason', '')}"

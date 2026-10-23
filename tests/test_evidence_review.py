import unittest
from pydantic import ValidationError
from backend.evidence_review import (VisualEvidence, TextEvidence, reconcile,
                                     visual_gate, render_evidence)


def base(action='limit', **kwargs):
    return dict(action=action, reason='候选视觉证据', text_violation=False,
                rule_id=None, **kwargs)


def negative():
    return dict(status='ok', injury='absent', attack='absent', threat='absent')


class ReviewTests(unittest.TestCase):
    def test_cartoon_visual_evidence_can_be_withdrawn(self):
        result = reconcile(base(), negative(), visual_called=True)
        self.assertEqual(result['action'], 'allow')
        self.assertFalse(result['review_hint'])

    def test_text_must_be_resolved_before_withdrawing_restriction(self):
        for text in [None, {'status': 'failed'}, {'status': 'ok', 'verdict': 'uncertain'}]:
            result = reconcile(base(), negative(), text, visual_called=True,
                               text_called=True, text_guard=True)
            self.assertEqual(result['action'], 'limit')
            self.assertTrue(result['review_hint'])

    def test_confirmed_threat_survives_negative_visual_review(self):
        text = dict(status='ok', verdict='violation', text='I WILL HURT YOU',
                    quote='I WILL HURT YOU')
        result = reconcile(base(), negative(), text, visual_called=True,
                           text_called=True, text_guard=True)
        self.assertEqual(result['action'], 'limit')
        self.assertTrue(result['text_violation'])
        self.assertFalse(result['review_hint'])

    def test_safe_watermark_removes_only_text_hint(self):
        text = dict(status='ok', verdict='safe', text='AI生成', quote='')
        result = reconcile(base('allow'), text_review=text, text_called=True, text_hint=True)
        self.assertFalse(result['review_hint'])
        result = reconcile(base(), None, text, visual_called=True, text_called=True)
        self.assertEqual(result['action'], 'limit')
        self.assertTrue(result['review_hint'])

    def test_supported_visual_review_does_not_change_penalty(self):
        for action in ['limit', 'block']:
            result = reconcile(base(action), dict(status='ok', injury='severe'), visual_called=True)
            self.assertEqual(result['action'], action)

    def test_existing_text_evidence_is_protected_and_not_rechecked(self):
        decision = base(); decision['text_violation'] = True
        self.assertFalse(visual_gate(decision))
        self.assertFalse(visual_gate(base('allow')))
        self.assertTrue(visual_gate(base()))
        self.assertEqual(reconcile(decision, negative(), visual_called=True)['action'], 'limit')

    def test_invalid_threat_relation_and_invented_quote_rejected(self):
        with self.assertRaises(ValidationError):
            VisualEvidence(injury='absent', attack='absent', threat='present',
                appearance='ordinary', threat_basis='none', threat_target='无',
                weapon_pose='持刀', reason='可能威胁')
        with self.assertRaises(ValidationError):
            TextEvidence(text='普通水印', verdict='violation', quote='I WILL HURT YOU', reason='威胁')

    def test_unclear_text_cannot_be_reported_safe(self):
        result = TextEvidence(text='[?] SHOOT', verdict='safe', quote='', reason='读不清')
        self.assertEqual(result.verdict, 'uncertain')

    def test_output_has_decision_and_no_rejected_raw_candidates(self):
        result = reconcile(base(), negative(), visual_called=True)
        rendered = render_evidence(result)
        self.assertIn('处置：放行', rendered)
        self.assertNotIn('候选', rendered)


if __name__ == '__main__':
    unittest.main()

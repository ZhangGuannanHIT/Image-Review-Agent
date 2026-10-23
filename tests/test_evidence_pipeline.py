import io
import json
import tempfile
import unittest
from PIL import Image
from langchain_core.messages import AIMessage
from backend.evidence_policy import combine
from backend.evidence_pipeline import EvidencePipeline
from backend.evidence_prompts import SEMANTIC_PROMPT, EXPLANATION_PROMPT
from backend.service import ReviewService
from src.labels import TYPE_CLASSES, INJURY_CLASSES, RISK_CLASSES


def evidence():
    single = lambda classes: {"scores": {x: .96 if i == 0 else .04/(len(classes)-1) for i,x in enumerate(classes)}}
    return {"trained": True, "risk": single(RISK_CLASSES), "injury": single(INJURY_CLASSES),
            "type": {"labels": [TYPE_CLASSES[0]], "scores": dict(zip(TYPE_CLASSES,[.9,.05,.05,.05])),
                     "thresholds": dict(zip(TYPE_CLASSES,[.45,.6,.6,.9]))},
            "quality": {"labels": ["清晰"]},
            "blood_binary": {"trained": True, "p_blood": .1, "threshold": .285}}


class Model:
    def bind(self, **kwargs):return self
    def invoke(self, messages):
        if messages[0].content == SEMANTIC_PROMPT:
            return AIMessage(content=json.dumps({"verdict":"violation","quote":"I WILL HURT YOU","reason":"明确人身威胁"}))
        # Deliberately tries to overwrite the program's action and hint.
        return AIMessage(content=json.dumps({"action":"allow","review_hint":True,"reason":"转人工"}))


class Engine:
    trained = True
    def predict(self, data):return evidence()


class OCR:
    def detect(self, data):return [[[0,0],[10,0],[10,10],[0,10]]]
    def recognize(self, data):return {"success":True,"text":"I WILL HURT YOU","texts":["I WILL HURT YOU"]}


class EvidenceTests(unittest.TestCase):
    def test_missing_detection_does_not_bypass_text_protection(self):
        class RestrictedEngine(Engine):
            def predict(self, data):
                item=evidence()
                item['type']['scores'][TYPE_CLASSES[0]]=.1
                item['type']['scores'][TYPE_CLASSES[1]]=.7
                return item
        class NoText(OCR):
            def detect(self, data):return []
            def recognize(self, data):raise AssertionError('No OCR gate')
        class Reviewer:
            def review_visual(self, data):
                return dict(status='ok', injury='absent', attack='absent', threat='absent')
            def review_text(self, data):
                return dict(status='ok', verdict='violation', text='I WILL HURT YOU', quote='I WILL HURT YOU')
        from backend.tools import set_current_image,reset_current_image
        token=set_current_image(b'test')
        try:
            result=EvidencePipeline(RestrictedEngine(),NoText(),Model(),
                evidence_reviewer=Reviewer(),rag_enabled=False).invoke({})
            self.assertEqual(result['decision']['action'],'limit')
            self.assertTrue(result['decision']['text_violation'])
            self.assertIn('review_text_evidence',result['tool_results'])
        finally:reset_current_image(token)

    def test_unverified_quote_and_tool_failure_cannot_upgrade(self):
        for semantic in [{"status":"failed","verdict":"violation","quote":"I WILL HURT YOU"},
                         {"status":"ok","verdict":"violation","quote":"invented text"}]:
            self.assertEqual(combine(evidence(),semantic=semantic,text="I WILL HURT YOU",text_gate=True)['action'],'allow')

    def test_literal_threat_but_not_photography_or_negation(self):
        self.assertEqual(combine(evidence(),text="PLEASE LOOT! I LOVE TO SHOOT",text_gate=True)['action'],'limit')
        for text in ['Do not say: please loot! I love to shoot.', 'Please loot the snacks. I love to shoot hoops.']:
            self.assertEqual(combine(evidence(),text=text,text_gate=True)['action'],'allow')

    def test_binary_negative_preserves_independent_violence(self):
        item=evidence();item['type']['scores'][TYPE_CLASSES[0]]=.1;item['type']['scores'][TYPE_CLASSES[1]]=.7
        self.assertEqual(combine(item)['action'],'limit')
        item['type']['scores'][TYPE_CLASSES[1]]=.1;item['type']['scores'][TYPE_CLASSES[2]]=.91
        self.assertEqual(combine(item)['action'],'limit')

    def test_untrained_clip_cannot_upgrade(self):
        item=evidence();item['trained']=False
        self.assertEqual(combine(item,text="PLEASE LOOT! I LOVE TO SHOOT",text_gate=True)['action'],'allow')

    def test_end_to_end_service_uses_text_and_rejects_action_override(self):
        engine,ocr,model=Engine(),OCR(),Model()
        pipeline=EvidencePipeline(engine,ocr,model,rag_enabled=False)
        buf=io.BytesIO();Image.new('RGB',(16,16),'white').save(buf,format='PNG')
        with tempfile.TemporaryDirectory() as directory:
            service=ReviewService(model,engine,ocr,log_dir=directory,evidence_pipeline=pipeline)
            response=service.review(buf.getvalue(),'example.png')
            self.assertIn('处置：限流',response.suggestion)
            self.assertNotIn('人工',response.suggestion)
            self.assertIn('文字',response.suggestion)
            from pathlib import Path
            record=json.loads((Path(directory)/'review.log.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(record['decision']['action'],'limit')
            self.assertIn('assess_text',record['tool_results'])

    def test_detector_failure_still_attempts_ocr(self):
        class BadDetector(OCR):
            def detect(self,data):raise RuntimeError('detector failed')
        from backend.tools import set_current_image,reset_current_image
        token=set_current_image(b'test')
        try:
            result=EvidencePipeline(Engine(),BadDetector(),Model(),rag_enabled=False).invoke({})
            self.assertEqual(result['decision']['action'],'limit')
            self.assertTrue(result['degraded'])
        finally:reset_current_image(token)

    def test_clip_failure_does_not_run_ocr(self):
        class BadEngine(Engine):
            def predict(self,data):raise RuntimeError('missing CLIP')
        from backend.tools import set_current_image,reset_current_image,get_request_context
        token=set_current_image(b'test')
        try:
            with self.assertRaises(RuntimeError):EvidencePipeline(BadEngine(),OCR(),Model()).invoke({})
            self.assertNotIn('ocr_image',get_request_context().cache)
        finally:reset_current_image(token)

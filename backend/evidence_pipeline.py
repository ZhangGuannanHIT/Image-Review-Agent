"""Controlled evidence workflow: successful CLIP first, gated tools, fixed action."""
import difflib
import json
import re
import threading
import time
from langchain_core.messages import HumanMessage, SystemMessage
from .tools import get_request_context, _as_dict
from .policy import DecisionPolicy
from .evidence_policy import combine
from .vision import VisionGate, trigger
from .rag import retrieve
from .evidence_prompts import SEMANTIC_PROMPT, RETRY_PROMPT
from .evidence_review import EvidenceReviewer, visual_gate, visual_verdict, reconcile, render_evidence


def normalized(text):
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', text.lower())


class EvidencePipeline:
    def __init__(self, engine, ocr, model, vision_client=None, vision_gate=None,
                 decision_policy=None, rag_enabled=True, evidence_reviewer=None):
        self.engine, self.ocr, self.vision_client = engine, ocr, vision_client
        self.gate = VisionGate(**(vision_gate or {}))
        self.policy = DecisionPolicy(**(decision_policy or {}))
        self.rag_enabled = rag_enabled
        self.reviewer = evidence_reviewer or EvidenceReviewer(model)
        self.model = model.bind(response_format={"type": "json_object"}, max_tokens=700,
                                extra_body={"thinking": {"type": "disabled"}})
        # A request uses both GPU models; serialize to keep memory and tool state bounded.
        self.lock = threading.RLock()

    def _json(self, prompt, data):
        response = self.model.invoke([SystemMessage(content=prompt),
            HumanMessage(content=json.dumps(data, ensure_ascii=False))])
        result = json.loads(response.content)
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        return result

    def semantic(self, sources, prompt=SEMANTIC_PROMPT, references=None):
        joined = '\n'.join(sources.values())
        if len(re.sub(r'\W', '', joined)) < 3:
            return {"status": "no_text", "verdict": "uncertain"}
        data = {"ocr_sources": sources}
        if references is not None:
            data["reference_rules"] = references
        result = self._json(prompt, data)
        valid = result.get("verdict") in {"safe", "violation", "uncertain"}
        quote = result.get("quote", "")
        if result.get("verdict") == "violation":
            valid = valid and isinstance(quote, str) and len(quote.strip()) >= 3 and any(quote in text for text in sources.values())
        result["status"] = "ok" if valid else "invalid"
        return result

    def invoke(self, state, config=None):
        with self.lock:
            return self._run()

    def _run(self):
        context = get_request_context()
        steps, results = [], {}

        def call(name, fn, required=False):
            start = time.monotonic()
            try:
                value = fn()
                results[name] = context.cache[name] = value
                ok = not isinstance(value, dict) or value.get("status") not in {"failed", "invalid"}
                steps.append({"type": "tool_call", "name": name, "ok": ok,
                              "elapsed_ms": round((time.monotonic()-start)*1000)})
                return value
            except Exception as exc:
                steps.append({"type": "tool_error", "name": name, "ok": False,
                              "error": type(exc).__name__})
                if required:
                    raise
                results[name] = context.cache[name] = {"status": "failed", "error": type(exc).__name__}
                return None

        evidence = call("analyze_image", lambda: _as_dict(self.engine.predict(context.image)), True)
        condition = trigger(evidence, self.gate)
        vision = None
        if self.vision_client and condition["eligible"]:
            vision = call("review_blood", lambda: {**self.vision_client.review(context.image), "trigger": condition})
        boxes = call("detect_text", lambda: self.ocr.detect(context.image))
        quality = evidence.get("quality", {}).get("labels", [])
        # Detector failure must not masquerade as proof that no text exists.
        text_gate = bool(boxes) or "文字扭曲形变" in quality or boxes is None
        text, semantic = "", {}
        if text_gate:
            recognized = call("ocr_image", lambda: self.ocr.recognize(context.image))
            text = (recognized or {}).get("text", "")
            hits = []
            if self.rag_enabled and len(re.sub(r'\W', '', text)) >= 3:
                hits = call("retrieve_text_rules", lambda: retrieve(text, k=5)) or []
            references = [{f: h[f] for f in ['id', 'content', 'negative_example'] if f in h} for h in hits]
            semantic = call("assess_text", lambda: self.semantic({"glm_local": text[:1800]}, references=references)) or {}
        decision = combine(evidence, self.policy, vision, semantic, text, text_gate)
        # Local OCR disagreement is evidence for a targeted text check.
        text_hint = False
        retry_needed = False
        if decision["action"] == "allow" and "文字扭曲形变" in quality and len(normalized(text)) >= 6:
            retry_needed = True
            sources = call("retry_text", lambda: self.ocr.retry_sources(context.image, boxes or [], text))
            if sources:
                assessment = call("assess_text_retry", lambda: self.semantic(sources, RETRY_PROMPT)) or {}
                other = [normalized(sources.get(mode, "")) for mode in ['contrast', 'crop']]
                similarity = min([difflib.SequenceMatcher(None, normalized(text), s).ratio()
                                  for s in other if len(s) >= 3], default=1)
                if similarity < .5 or (assessment.get("status") == "ok" and assessment.get("verdict") == "uncertain"):
                    text_hint = True
        visual_called = visual_gate(decision)
        visual = call("review_visual_evidence", lambda: self.reviewer.review_visual(context.image)) if visual_called else None
        # Never withdraw a restriction on an image with possible text until the text is checked.
        # Small handwritten threats can be missed by both detector and quality head.
        text_guard = visual_called and visual_verdict(visual) == "not_supported"
        text_called = retry_needed or text_guard
        text_review = call("review_text_evidence", lambda: self.reviewer.review_text(context.image)) if text_called else None
        decision = reconcile(decision, visual, text_review, visual_called=visual_called,
                             text_called=text_called, text_guard=text_guard, text_hint=text_hint)
        suggestion = render_evidence(decision)
        return {"decision": decision, "suggestion": suggestion, "clip_called": True,
                "clip_succeeded": True, "finalized": True, "degraded": any(not s['ok'] for s in steps),
                "steps": steps, "tool_results": results, "messages": []}

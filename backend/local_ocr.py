"""Local scene-text OCR. No network downloads or image interpretation endpoint."""
from pathlib import Path
import io
import threading

import numpy as np
from PIL import Image, ImageOps, ImageFilter


class LocalSceneOCR:
    def __init__(self, model_dir, rapid_dir, easy_dir, device="cuda"):
        import torch
        from rapidocr import RapidOCR
        from transformers import AutoProcessor, GlmOcrForConditionalGeneration
        self.lock = threading.RLock()
        self.device = device
        self.easy_dir = str(easy_dir)
        self.rapid_dir = str(rapid_dir)
        self.easy = self.large = None
        self.rapid = RapidOCR(params=self._rapid_params())
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = GlmOcrForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            local_files_only=True).to(device).eval()
        self.processor.image_processor.size = {"shortest_edge": 12544, "longest_edge": 2*1280*1280}

    def _rapid_params(self, large=False):
        return {"Global.model_root_dir": self.rapid_dir, "Global.log_level": "error",
                "EngineConfig.onnxruntime.intra_op_num_threads": 4,
                "EngineConfig.onnxruntime.inter_op_num_threads": 2,
                "Det.limit_side_len": 1280 if large else 736}

    @staticmethod
    def image(data):
        if isinstance(data, Image.Image):
            return data.convert("RGB")
        with Image.open(io.BytesIO(data)) as im:
            return im.convert("RGB")

    def detect(self, data):
        with self.lock:
            result = self.rapid(np.asarray(self.image(data)), use_cls=False, use_rec=False)
            return result.boxes.tolist() if result.boxes is not None else []

    def recognize(self, data):
        import torch
        with self.lock:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": self.image(data)},
                {"type": "text", "text": "Text Recognition:"}]}]
            inputs = self.processor.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=192, do_sample=False)
            tail = output[0, inputs["input_ids"].shape[-1]:]
            text = self.processor.decode(tail, skip_special_tokens=True)
            return {"success": True, "status": "ok", "provider": "local",
                    "model": "GLM-OCR", "texts": [text] if text else [],
                    "text": text, "truncated": len(tail) >= 192}

    def retry_sources(self, data, boxes, original):
        """Only called for unresolved distorted text, using the tested five readings."""
        from rapidocr import RapidOCR
        import easyocr
        with self.lock:
            im = self.image(data)
            contrast = ImageOps.autocontrast(ImageOps.grayscale(im), cutoff=1).filter(
                ImageFilter.UnsharpMask(radius=1.5, percent=140, threshold=3)).convert("RGB")
            points = [point for box in boxes for point in box]
            crop = im
            if points:
                xs, ys = zip(*points)
                pad = max(12, int(min(im.size)*.025))
                crop = im.crop((max(0, int(min(xs))-pad), max(0, int(min(ys))-pad),
                                min(im.width, int(max(xs))+pad), min(im.height, int(max(ys))+pad)))
            if self.large is None:
                self.large = RapidOCR(params=self._rapid_params(large=True))
            if self.easy is None:
                self.easy = easyocr.Reader(['en'], gpu=self.device.startswith('cuda'),
                    model_storage_directory=self.easy_dir, user_network_directory=self.easy_dir,
                    download_enabled=False, verbose=False)
            rapid = self.large(np.asarray(im))
            easy = self.easy.readtext(np.asarray(im), detail=1, paragraph=False,
                canvas_size=2560, mag_ratio=1.5, batch_size=8, workers=0)
            return {"original": original, "rapid": "\n".join(rapid.txts or []),
                    "easy": "\n".join(str(row[1]) for row in easy),
                    "contrast": self.recognize(contrast)["text"],
                    "crop": self.recognize(crop)["text"]}

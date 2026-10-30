"""Lightweight, CPU-only text-region detection for OCR gating.

This detector does not read text. It uses edge structure and horizontal
morphology to decide whether a full OCR call is worth running. The operating
point intentionally favours recall: a false positive only adds an OCR call,
while a false negative can hide a textual threat.
"""

from __future__ import annotations

import io
from typing import Any, Union

import cv2
import numpy as np
from PIL import Image, ImageOps


ImageInput = Union[bytes, bytearray, Image.Image]


def _decode_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(bytes(image))) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    elif isinstance(image, Image.Image):
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    else:
        raise TypeError(f"unsupported image type: {type(image)!r}")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _resize_for_detection(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = bgr.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr, scale


class LightweightTextDetector:
    """High-recall text-line candidate detector based on OpenCV."""

    method = "opencv_text_regions_v1"

    def __init__(
        self,
        max_side: int = 1600,
        min_line_height: int = 7,
        min_layout_score: float = 0.48,
        min_region_score: float = 0.52,
        min_aligned_glyphs: int = 2,
    ):
        if max_side < 256:
            raise ValueError("max_side must be at least 256")
        self.max_side = int(max_side)
        self.min_line_height = int(min_line_height)
        self.min_layout_score = float(min_layout_score)
        self.min_region_score = float(min_region_score)
        self.min_aligned_glyphs = int(min_aligned_glyphs)

    @staticmethod
    def _glyph_layout_score(region: np.ndarray) -> tuple[float, int]:
        """Score a candidate by repeated, similarly sized glyph components."""
        if region.size < 64:
            return 0.0, 0
        block = max(15, min(41, (region.shape[1] // 3) | 1))
        masks = [
            cv2.adaptiveThreshold(
                region,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block,
                9,
            ),
            cv2.adaptiveThreshold(
                region,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                9,
            ),
        ]
        best_score = 0.0
        best_count = 0
        line_height = region.shape[0]
        line_width = region.shape[1]
        for mask in masks:
            count, _, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            glyphs = []
            for index in range(1, count):
                x, y, width, height, area = stats[index]
                if width < 2 or height < 4:
                    continue
                if height < max(4, int(line_height * 0.22)):
                    continue
                if height > line_height * 1.08 or width > line_width * 0.55:
                    continue
                if area < max(8, int(line_height * line_height * 0.015)):
                    continue
                if width / max(height, 1) > 4.5:
                    continue
                glyphs.append((x, y, width, height, area, centroids[index]))
            if len(glyphs) < 2:
                continue
            heights = np.asarray([item[3] for item in glyphs], dtype=float)
            median_height = float(np.median(heights))
            if median_height < 4:
                continue
            aligned = [
                item for item in glyphs
                if 0.55 * median_height <= item[3] <= 1.65 * median_height
            ]
            if len(aligned) < 2:
                continue
            centers = np.asarray([item[5][1] for item in aligned], dtype=float)
            center_spread = float(np.median(np.abs(centers - np.median(centers))))
            if center_spread > median_height * 0.8:
                continue
            ordered = sorted(aligned, key=lambda item: item[0])
            gaps = [
                max(0, right[0] - (left[0] + left[2]))
                for left, right in zip(ordered, ordered[1:])
            ]
            if gaps and float(np.median(gaps)) > median_height * 3.0:
                continue
            glyph_count = len(aligned)
            score = min(
                0.99,
                0.30
                + min(glyph_count, 10) * 0.055
                + max(0.0, 0.12 - center_spread / max(median_height, 1)),
            )
            if score > best_score:
                best_score, best_count = score, glyph_count
        return float(best_score), best_count

    def detect(self, image: ImageInput) -> dict[str, Any]:
        bgr, scale = _resize_for_detection(_decode_bgr(image), self.max_side)
        height, width = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        # Horizontal gradients preserve character strokes; the wide close joins
        # characters into line-shaped regions without merging unrelated rows.
        gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        magnitude = cv2.convertScaleAbs(gradient)
        threshold, binary = cv2.threshold(
            magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # If Otsu picks an almost solid mask, use a quantile fallback.
        if float(np.mean(binary > 0)) > 0.45:
            cutoff = float(np.quantile(magnitude, 0.80))
            binary = np.where(magnitude >= cutoff, 255, 0).astype(np.uint8)
        else:
            cutoff = float(threshold)

        line_width = max(9, min(31, width // 70))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, 3))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        connected = cv2.morphologyEx(
            connected,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes: list[list[float]] = []
        scores: list[float] = []
        image_area = float(width * height)
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = float(box_width * box_height)
            if area < 32 or area > image_area * 0.45:
                continue
            if box_height < self.min_line_height or box_height > height * 0.35:
                continue
            if box_width < 12 or box_width > width * 0.98:
                continue
            aspect = box_width / max(box_height, 1)
            if aspect < 1.15 or aspect > 35.0:
                continue

            region = gray[y:y + box_height, x:x + box_width]
            if region.size == 0:
                continue
            edge = cv2.Canny(region, 60, 160)
            edge_ratio = float(np.mean(edge > 0))
            low, high = np.quantile(region, [0.05, 0.95])
            dark_cutoff = float(low + max(8.0, (high - low) * 0.35))
            dark_ratio = float(np.mean(region <= dark_cutoff))
            if edge_ratio < 0.025 or dark_ratio < 0.04:
                continue
            layout_score, glyph_count = self._glyph_layout_score(region)
            if (
                glyph_count < self.min_aligned_glyphs
                or layout_score < self.min_layout_score
            ):
                continue

            # Normalize back to the requested image coordinates.
            inv = 1.0 / scale if scale else 1.0
            boxes.append([
                max(0.0, x * inv / max(width * inv, 1.0)),
                max(0.0, y * inv / max(height * inv, 1.0)),
                min(1.0, (x + box_width) * inv / max(width * inv, 1.0)),
                min(1.0, (y + box_height) * inv / max(height * inv, 1.0)),
            ])
            scores.append(min(
                0.99,
                layout_score * 0.75
                + min(0.20, edge_ratio * 0.45)
                + min(aspect, 8) * 0.008,
            ))

        # Keep the strongest regions and avoid a noisy single region triggering
        # OCR unless it has meaningful edge support.
        ranked = sorted(zip(scores, boxes), reverse=True)[:24]
        strong = [
            (score, box) for score, box in ranked
            if score >= self.min_region_score
        ]
        if strong:
            ranked = strong
        boxes = [box for _, box in ranked]
        confidence = max((score for score, _ in ranked), default=0.0)
        return {
            "status": "detected" if boxes else "empty",
            "has_text": bool(boxes),
            "boxes": boxes,
            "box_count": len(boxes),
            "confidence": round(float(confidence), 4),
            "method": self.method,
            "gradient_cutoff": round(cutoff, 2),
        }


class RapidOCRTextDetector:
    """Optional detector backend when a local RapidOCR model is available."""

    method = "rapidocr_detection_v1"

    def __init__(self, model_root_dir: str = ""):
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "RapidOCR 未安装；请安装 rapidocr 或改用 opencv_light"
            ) from exc
        params = {"Global.log_level": "error"}
        if model_root_dir:
            params["Global.model_root_dir"] = model_root_dir
        self.engine = RapidOCR(params=params)

    def detect(self, image: ImageInput) -> dict[str, Any]:
        rgb = _decode_bgr(image)
        result = self.engine(rgb, use_cls=False, use_rec=False)
        boxes = result.boxes.tolist() if result.boxes is not None else []
        return {
            "status": "detected" if boxes else "empty",
            "has_text": bool(boxes),
            "boxes": boxes,
            "box_count": len(boxes),
            "confidence": 1.0 if boxes else 0.0,
            "method": self.method,
        }

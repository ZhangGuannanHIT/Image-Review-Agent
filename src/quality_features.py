# -*- coding: utf-8 -*-
"""Deterministic low-level features for blur, occlusion, and image quality."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image


FEATURE_NAMES = [
    "laplacian_log",
    "laplacian_p90_log",
    "tenengrad_log",
    "edge_density",
    "fft_high_freq_ratio",
    "entropy",
    "contrast_std",
    "dark_frac_20",
    "dark_frac_40",
    "largest_dark_component_ratio",
    "largest_dark_bbox_fill",
    "blockiness_ratio",
    "local_detail_log",
    "saturation_mean",
    "saturation_std",
]


def _resize_for_features(image: Image.Image, max_side: int = 1024) -> np.ndarray:
    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        rgb = rgb.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(rgb, dtype=np.uint8)


def extract_quality_features(image: Image.Image) -> np.ndarray:
    rgb = _resize_for_features(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    laplacian_log = math.log1p(float(laplacian_abs.var()))
    laplacian_p90_log = math.log1p(float(np.percentile(laplacian_abs, 90)))

    grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = grad_x * grad_x + grad_y * grad_y
    tenengrad_log = math.log1p(float(tenengrad.mean()))

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(
        cv2.resize(gray_float, (256, 256), interpolation=cv2.INTER_AREA)
    )))
    center = spectrum.shape[0] // 4
    low = spectrum[
        spectrum.shape[0] // 2 - center:spectrum.shape[0] // 2 + center,
        spectrum.shape[1] // 2 - center:spectrum.shape[1] // 2 + center,
    ]
    total_energy = float(spectrum.sum())
    fft_high_freq_ratio = (
        1.0 - float(low.sum()) / total_energy if total_energy > 0 else 0.0
    )

    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(histogram.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    contrast_std = float(gray_float.std())

    dark_20 = gray < 20
    dark_40 = gray < 40
    dark_frac_20 = float(dark_20.mean())
    dark_frac_40 = float(dark_40.mean())

    dark_mask = (dark_40.astype(np.uint8) * 255)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark_mask, 8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.max())
        best = 1 + int(areas.argmax())
        x = stats[best, cv2.CC_STAT_LEFT]
        y = stats[best, cv2.CC_STAT_TOP]
        width = stats[best, cv2.CC_STAT_WIDTH]
        height = stats[best, cv2.CC_STAT_HEIGHT]
        largest_dark_component_ratio = largest / float(gray.size)
        largest_dark_bbox_fill = largest / float(max(width * height, 1))
        # Prefer large, compact rectangular regions typical of black bars.
        largest_dark_component_ratio *= min(1.0, largest_dark_bbox_fill / 0.65)
        _ = (x, y)
    else:
        largest_dark_component_ratio = 0.0
        largest_dark_bbox_fill = 0.0

    diff_x = np.abs(np.diff(gray_float, axis=1))
    diff_y = np.abs(np.diff(gray_float, axis=0))
    boundary_x = diff_x[:, 7::8].mean() if diff_x.shape[1] > 8 else 0.0
    boundary_y = diff_y[7::8, :].mean() if diff_y.shape[0] > 8 else 0.0
    interior_x = np.delete(diff_x, np.s_[7::8], axis=1).mean()
    interior_y = np.delete(diff_y, np.s_[7::8], axis=0).mean()
    boundary = float(boundary_x + boundary_y)
    interior = float(interior_x + interior_y)
    blockiness_ratio = boundary / max(interior, 1e-6)

    residual = gray_float - cv2.GaussianBlur(gray_float, (0, 0), 1.2)
    local_detail_log = math.log1p(float(np.abs(residual).mean() * 255.0))

    vector = np.asarray([
        laplacian_log,
        laplacian_p90_log,
        tenengrad_log,
        edge_density,
        fft_high_freq_ratio,
        entropy,
        contrast_std,
        dark_frac_20,
        dark_frac_40,
        largest_dark_component_ratio,
        largest_dark_bbox_fill,
        blockiness_ratio,
        local_detail_log,
        float(saturation.mean()),
        float(saturation.std()),
    ], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("quality features 包含非有限值")
    return vector

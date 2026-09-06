# -*- coding: utf-8 -*-
"""低层像素特征提取器：在 CLIP 语义判断之外，用像素统计兜底检异常。

定位：**可疑度信号 / 分流依据**，绝不替代 CLIP 分类头做裁决。
纯 OpenCV/numpy 计算，1~3ms/张，纯 CPU，不依赖标注、不膨胀 prompt、不违背立项。

【能力边界（重要）】低层特征擅长捕捉"全局像素规律"，但不擅长"内容语义"：
  - red_ratio        红色占比（HSV 红区）→ 疑似大片红（注意：红旗/番茄酱也红，仅提示）
  - blur_score       拉普拉斯方差 → 越低越模糊（整图模糊/打码）
  - edge_density     Canny 边缘密度 → 大面积马赛克/密集纹理
  - text_region_ratio 文字弱提示（仅在全图文字密集时提示，**局部细小水印检不出，交 OCR**）
  - entropy          灰度信息熵 → 异常压缩/截图/劣化

【局限】低层特征分不清"文字"与"斑驳纹理"（都是小连通域），也认不出"局部小水印"。
      因此文字识别【一律交给 OCR 工具（ocr_image）】，本模块的 text_region_ratio
      仅作为"候选触发 OCR"的弱信号，绝不据此判"有文字"或"违规"。

阈值均为初始猜测，需用少量真实样本标定（见 extract 内注释）。
"""

from __future__ import annotations

import io
from typing import Union

import numpy as np
import cv2
from PIL import Image

# 初始阈值（可在 _THRESHOLDS 整体调整标定）
_THRESHOLDS = {
    "red_ratio": 0.25,         # 红色占比超此为"疑似血腥"提示
    "blur_score": 80.0,        # 拉普拉斯方差低于此为"较模糊"
    "edge_density": 0.12,      # 边缘密度超此为"密集纹理/疑似打码"
    "text_region_ratio": 0.30, # 文字密集度弱提示阈值（边缘密度代理；仅在"全图文字密集"才提示，交 OCR；宁少报）
    "entropy": 5.0,            # 信息熵低于此为"疑似过度压缩/截图"
}

# 红色 HSV 上下界（H 环，0 附近与 180 附近都是红）
_RED_HSV = [
    ((0, 60, 60), (10, 255, 255)),
    ((170, 60, 60), (180, 255, 255)),
]


def _to_bgr(image: Union[bytes, "Image.Image", np.ndarray], target: int = 256) -> np.ndarray:
    """把 bytes / PIL / ndarray 统一转成 OpenCV BGR 图，并缩到 target 边长（提速）。"""
    if isinstance(image, bytes):
        arr = np.frombuffer(image, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            # 尝试用 PIL 兜底解码（含 EXIF 方向的 WEBP 等）
            pil = Image.open(io.BytesIO(image)).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        bgr = image.copy()
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("ndarray 图片必须是 HxWx3")
        if bgr.dtype != np.uint8:
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    else:  # PIL Image
        rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    s = target / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def extract(image: Union[bytes, "Image.Image", np.ndarray]) -> dict:
    """提取低层特征，返回结构化 dict（浮点，0~1 或绝对量）。

    纯计算，不做任何判罚，只给"特征值"。
    文字信号用原始分辨率计算（识别"全图文字密集"），其余特征用缩图（提速）。
    """
    bgr = _to_bgr(image, target=256)     # 缩图，用于 red/blur/edge/entropy
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    total = float(gray.size)

    # 1) 红色占比（血腥）
    red_mask = np.zeros_like(gray, dtype=np.uint8)
    for lo, hi in _RED_HSV:
        red_mask |= cv2.inRange(hsv, lo, hi)
    red_ratio = float((red_mask > 0).sum()) / total

    # 2) 模糊度（拉普拉斯方差，越低越模糊）
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 3) 边缘密度（马赛克/打码/密集纹理）
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).sum()) / total

    # 4) 文字区域占比（弱信号，用原始分辨率，识别"全图文字密集"）
    #    说明：低层特征分不清"文字"与"斑驳纹理"（都是小连通域），也认不出局部小水印。
    #         此信号仅在"全图文字密集"时提示，真正的文字识别交给 OCR 工具（ocr_image）。
    text_region_ratio = _text_region_ratio(image)

    # 5) 信息熵（异常压缩/截图）
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    if entropy < 0:
        entropy = 0.0  # 避免浮点负零

    return {
        "red_ratio": round(red_ratio, 4),
        "blur_score": round(blur_score, 2),
        "edge_density": round(edge_density, 4),
        "text_region_ratio": round(text_region_ratio, 4),
        "entropy": round(entropy, 3),
    }


def _text_region_ratio(image: Union[bytes, "Image.Image", np.ndarray]) -> float:
    """文字密集度代理（弱信号）：用"原始分辨率下的边缘密度"近似。

    文字/细密纹理会产生大量边缘；相比连通域法（过密会粘连成一块、过疏会漏），
    边缘密度更稳定。仅用于"全图文字密集"的弱提示，局部小水印交给 OCR。
    """
    full = _to_bgr(image, target=768)
    gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
    total = float(gray.size)
    edges = cv2.Canny(gray, 80, 180)
    return float((edges > 0).sum()) / total


def to_text(feats: dict) -> str:
    """把特征转成 LLM 可直接读的可读文本，仅输出"提示"不输出结论。"""
    flags: list[str] = []
    if feats["red_ratio"] > _THRESHOLDS["red_ratio"]:
        flags.append(f"红色占比高({feats['red_ratio']:.2f})，疑似血腥/红色醒目元素")
    if feats["blur_score"] < _THRESHOLDS["blur_score"]:
        flags.append(f"画面较模糊(方差{feats['blur_score']:.0f})")
    if feats["edge_density"] > _THRESHOLDS["edge_density"]:
        flags.append(f"边缘密度高({feats['edge_density']:.2f})，疑似打码/密集纹理")
    if feats["text_region_ratio"] > _THRESHOLDS["text_region_ratio"]:
        flags.append(f"画面细节/文字密集度异常高({feats['text_region_ratio']:.2f})，疑似含大量文字，建议 OCR 确认")
    if feats["entropy"] < _THRESHOLDS["entropy"]:
        flags.append(f"信息熵偏低({feats['entropy']:.2f})，疑似过度压缩/截图")
    if not flags:
        return "低层特征未见明显异常（红色/模糊/边缘/文字/熵均正常）"
    return "低层特征提示: " + "；".join(flags)


# ---- 路由辅助：仅供 Router 前置分流，不参与 LLM 裁决 ----
def suspicion_score(feats: dict) -> float:
    """汇总一个 [0,1] 疑度分：命中提示越多越高。仅供分流，不做最终判定。"""
    score = 0.0
    if feats["red_ratio"] > _THRESHOLDS["red_ratio"]:
        score += 0.25
    if feats["blur_score"] < _THRESHOLDS["blur_score"]:
        score += 0.25
    if feats["edge_density"] > _THRESHOLDS["edge_density"]:
        score += 0.2
    if feats["text_region_ratio"] > _THRESHOLDS["text_region_ratio"]:
        score += 0.2
    if feats["entropy"] < _THRESHOLDS["entropy"]:
        score += 0.1
    return round(min(score, 1.0), 3)


def has_much_text(feats: dict) -> bool:
    """是否为"疑似存在文字"（弱提示，用于候选触发 OCR）。
    注意：低层特征无法可靠判定文字，这里仅在特征明显时才返回 True，真正识别交给 OCR。"""
    return feats["text_region_ratio"] > _THRESHOLDS["text_region_ratio"]


if __name__ == "__main__":
    # 自检：用一张纯色图 + 一张示例图验证不报错
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else None
    if sample:
        with open(sample, "rb") as f:
            feats = extract(f.read())
    else:
        bgr = np.zeros((200, 200, 3), np.uint8)
        bgr[:, :] = (0, 0, 255)  # 纯红（BGR）
        feats = extract(bgr)
    print(feats)
    print(to_text(feats))
    print("suspicion:", suspicion_score(feats))

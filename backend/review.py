# -*- coding: utf-8 -*-
"""【废弃】旧固定流程编排。已被 service.py（ReAct agent 版）取代，保留仅为引用兼容。

如无外部引用，可直接删除本文件。新编排见 backend/service.py。
"""

from __future__ import annotations

import warnings

warnings.warn("backend.review 已废弃，请改用 backend.service（ReAct agent）。", DeprecationWarning)


class ReviewOrchestrator:
    def __init__(self, service=None, *args, **kwargs):
        self.service = service

    def review(self, image_bytes, filename=""):
        if self.service is None:
            raise RuntimeError("旧 ReviewOrchestrator 需要传入 ReviewService；新代码请直接使用 ReviewService")
        return self.service.review(image_bytes, filename)


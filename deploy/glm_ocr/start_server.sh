#!/usr/bin/env bash
set -euo pipefail

PYTHON="${GLMOCR_PYTHON:-/home/ygx/miniconda3/envs/glmocr/bin/python}"
CONFIG="${GLMOCR_CONFIG:-/mnt/d/Image-Review-Agent/deploy/glm_ocr/config.yaml}"

exec "$PYTHON" -m glmocr.server --config "$CONFIG"

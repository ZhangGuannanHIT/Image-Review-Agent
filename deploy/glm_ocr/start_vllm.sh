#!/usr/bin/env bash
set -euo pipefail

PYTHON="${GLMOCR_PYTHON:-/home/ygx/miniconda3/envs/glmocr/bin/python}"
MODEL_DIR="${GLMOCR_MODEL_DIR:-/home/ygx/models/GLM-OCR}"

exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name glm-ocr \
  --host 127.0.0.1 \
  --port 8080 \
  --gpu-memory-utilization 0.46 \
  --max-model-len 6144 \
  --enforce-eager \
  --trust-remote-code

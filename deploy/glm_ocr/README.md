# Local GLM-OCR

This deployment uses an isolated WSL conda environment named `glmocr`.
Model weights stay outside Git:

- `/home/ygx/models/GLM-OCR`
- `/home/ygx/models/PP-DocLayoutV3_safetensors`

## Recreate environment

```bash
/home/ygx/miniconda3/bin/conda create -n glmocr python=3.12 pip -y
/home/ygx/miniconda3/envs/glmocr/bin/pip install vllm==0.19.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /mnt/d/GLM-OCR-src/GLM-OCR-main
/home/ygx/miniconda3/envs/glmocr/bin/pip install -e \
  '.[selfhosted,server]' -i https://pypi.tuna.tsinghua.edu.cn/simple
/home/ygx/miniconda3/envs/glmocr/bin/pip install modelscope \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
/home/ygx/miniconda3/envs/glmocr/bin/modelscope download \
  --model ZhipuAI/GLM-OCR --local_dir /home/ygx/models/GLM-OCR
/home/ygx/miniconda3/envs/glmocr/bin/modelscope download \
  --model PaddlePaddle/PP-DocLayoutV3_safetensors \
  --local_dir /home/ygx/models/PP-DocLayoutV3_safetensors
```


## Start

Use one WSL terminal for the moderation backend and one for GLM-OCR vLLM.

Terminal 1, vLLM model service:

```bash
bash /mnt/d/Image-Review-Agent/deploy/glm_ocr/start_vllm.sh
```

The moderation backend uses the single-image vLLM endpoint directly:

```text
http://127.0.0.1:8080/v1/chat/completions
```

This avoids the document-layout pipeline issuing one model request per text
region. The optional document pipeline can be started separately when structured
document layout is needed:

```bash
bash /mnt/d/Image-Review-Agent/deploy/glm_ocr/start_server.sh
```

Endpoints:

- vLLM OpenAI API: `http://127.0.0.1:8080/v1`
- GLM-OCR SDK API: `http://127.0.0.1:5002/glmocr/parse`

The moderation backend selects `local_glm_ocr` from `config/ocr.yaml` and
does not fall back to the cloud OCR provider. It uses `direct_vllm` mode by
default; `api_mode: pipeline` can point at port `5002` for document parsing.

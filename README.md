# ImageGuard 运行说明

以下命令在仓库根目录执行，示例环境为 Windows PowerShell、Python 3.12、NVIDIA GPU。
完成配置后，运行 `run_backend.py` 即可通过同一服务访问前端与后端；也支持前后端独立启动。

## 1. 安装依赖

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

需要支持 CUDA 12.8 的 NVIDIA 驱动。使用 CPU 时，将 PyTorch 安装命令中的 `cu128` 换成 `cpu`，并把 `backend/config.yaml` 中的 `model.device` 改为 `cpu`。

## 2. 放置已有基模

将已有的 **FG-CLIP2 Base** 放到仓库根目录的 `fg_clip/`，保留完整模型目录，包括：

```text
fg_clip/
  model.safetensors
  config.json
  configuration_fgclip2.py
  modeling_fgclip2.py
  preprocessor_config.json
  tokenizer.json
  tokenizer.model
  tokenizer_config.json
  special_tokens_map.json
```

训练后的 LoRA、分类头及标签定义已包含在 `checkpoints/best/`，不需要另外训练。

## 3. 配置 API 密钥

首次配置时复制模板；已有配置不要覆盖：

```powershell
Copy-Item config/llm.example.yaml config/llm.yaml
```

填写 `config/llm.yaml` 中的 `api_key`；也可通过 `LLM_API_KEY` 环境变量覆盖。文字语义判断、血腥核验、视觉关系核验和专项读字共用该配置，所选模型及 API 必须支持图片输入。实际密钥文件已被 Git 忽略。

首次准备本地 OCR 模型（需要联网；已有完整模型时可跳过）：

```powershell
.\.venv\Scripts\python.exe setup_local_models.py
.\.venv\Scripts\python.exe setup_local_models.py --check
```

脚本从 [GLM-OCR 官方仓库](https://huggingface.co/zai-org/GLM-OCR) 下载固定版本，并使用 RapidOCR、EasyOCR 自带下载器准备其模型。也可以将以下目录从已有环境复制过来。模型文件不上传 GitHub。

```text
local_models/glm_ocr/       # 完整 GLM-OCR 模型、处理器与 tokenizer，权重约 2.65 GB
local_models/rapidocr/      # PP-OCRv6_det_small.onnx、PP-OCRv6_rec_small.onnx、ch_ppocr_mobile_v2.0_cls_mobile.onnx
local_models/easyocr/       # craft_mlt_25k.pth、english_g2.pth
checkpoints/best/injury_binary/  # head.pt、metadata.json
checkpoints/best/quality_fusion/ # head.pt、metadata.json
```

请在启动前完成模型准备。路径在 `backend/config.yaml` 的 `evidence_pipeline` 中配置。

## 4. 启动后端

```powershell
.\.venv\Scripts\python.exe run_backend.py
```

默认端口 `8000`，健康检查地址：<http://127.0.0.1:8000/health>。返回 `service_ready: true`、`model_loaded: true`、`mock: false` 表示真实模型已就绪。使用 Ctrl+C 停止后端。

## 5. 打开前端

启动后端后，浏览器打开 <http://127.0.0.1:8000/> 即可使用审核工作台。

也可以独立启动前端：

另开一个终端，在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

浏览器打开 <http://127.0.0.1:5500/>，或自行使用 Live Server 等 HTTP 服务启动 `frontend/`。不要直接双击 HTML，`file://` 仅提供界面预览。前端后端地址在 `frontend/config.js` 中设置，默认 `http://127.0.0.1:8000/`；后端允许本机 localhost/127.0.0.1 的 HTTP 前端跨端口访问。

更换后端端口时，同时修改 `backend/config.yaml` 和 `frontend/config.js`。前端记录仅存当前页面，刷新会清空；后端原图和调用记录保存在 `backend/logs/`。

## 可调整的运行配置

- `config/inference.yaml`：CLIP 类型、图像质量阈值。
- `backend/config.yaml` → `evidence_pipeline.enabled: true`：启用最终证据流程。CLIP 先识别，按需检测文字、OCR、检索本地规则和核验语义。
- 视觉证据导致限流或下架、且没有独立文字违规证据时，触发视觉关系工具，分别核对伤害、攻击和武器指向对象。排除视觉证据前统一核对文字，防止检测漏框造成文字威胁漏放；疑难 OCR 也会触发专项读字。
- `vision_review.enabled`：前置血腥复核开关。该前置工具只在“无暴力”且有血腥候选、分数低于 `0.65` 或前两名分差小于 `0.20` 时触发；不控制上述视觉关系工具。
- 每次审核给出放行、限流或下架。关键证据无法核实时才附加人工核对提示；原始分数、工具输出和原图保存在 `backend/logs/`。
- `audit.save_images`：是否保存原图。日志与图片不上传 GitHub。
- `features.rag_enabled`：本地知识库检索开关。`knowledge/` 已随仓库提供，无需启动额外数据库或服务。

配置修改后重启后端。

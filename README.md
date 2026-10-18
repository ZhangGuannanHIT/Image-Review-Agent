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
Copy-Item config/ocr.example.yaml config/ocr.yaml
```

填写两个文件中的 `api_key`：

- `config/llm.yaml`：DeepSeek，裁决与血腥视觉复核共用此 key。
- `config/ocr.yaml`：GLM-OCR。

也可在启动终端设置 `LLM_API_KEY`、`OCR_API_KEY` 环境变量覆盖密钥，但仍需保留上述配置文件。实际密钥文件已被 Git 忽略。

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

- `config/inference.yaml`：已标定的暴力类型、图像质量阈值。
- `backend/config.yaml` → `vision_review.enabled`：血腥视觉复核开关。
- 复核仅在暴力类型只命中“无暴力”、血腥候选不是“无血腥”，且血腥分数低于 `0.65` 或前两名分差小于 `0.20` 时触发。每次请求最多调用一次，前端只显示复核结果。
- `audit.save_images`：是否保存审核原图。日志与图片不上传 GitHub。

配置修改后重启后端。

## 本地 RAG

已实现字符 n-gram TF-IDF 检索，无需额外模型或向量数据库。知识库包含 15 条规则、4 条核验案例、1 条教学案例。
Agent 成功取得视觉证据后，可按需查询相关规则；最终处置继续使用现有程序规则。
55 条查询用于阈值校准，不代表独立测试或真实图片审核准确率。

详细说明见 [RAG 与前端实现说明](docs/RAG与前端实现说明.md)。

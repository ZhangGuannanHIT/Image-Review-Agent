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

- `config/llm.yaml`：DeepSeek，用于最终解释与可选的血腥视觉复核。
- `config/ocr.yaml`：OCR。默认接入阿里云高精版 AppCode 接口，旧的 GLM-OCR 仍兼容。

也可在启动终端设置 `LLM_API_KEY`、`ALIYUN_OCR_APPCODE`（旧 GLM Provider 使用
`OCR_API_KEY`）环境变量覆盖密钥，但仍需保留上述配置文件。实际密钥文件已被 Git 忽略。

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
- `backend/config.yaml` → `features.llm_explanation_enabled`：程序裁决后由 LLM 生成解释；关闭或解释校验失败时使用确定性模板。
- `backend/config.yaml` → `text_detection`：轻量文字检测参数；默认 `opencv_light`，无需模型权重。
- `backend/config.yaml` → `anime_vision_review`：仅动漫高置信路由且动作非放行时调用视觉 LLM。
- 复核仅在暴力类型只命中“无暴力”、血腥候选不是“无血腥”，且血腥分数低于 `0.65` 或前两名分差小于 `0.20` 时触发。每次请求最多调用一次，前端只显示复核结果。
- `audit.save_images`：是否保存审核原图。日志与图片不上传 GitHub。

阿里云 OCR 配置见 [ALIYUN_OCR.md](docs/ALIYUN_OCR.md)。AppCode 只放在本地
`config/ocr.yaml` 或 `ALIYUN_OCR_APPCODE` 环境变量中。

文字扭曲 OCR 小样本结果见
[OCR_TEXT_DISTORTION_PILOT_2026-10-22.md](docs/OCR_TEXT_DISTORTION_PILOT_2026-10-22.md)。

开放词汇场景概念作为受控旁路工具，只在 CLIP 风险非 L0 时启用，不改变分类头：

```text
config/scene_concepts.yaml
knowledge/scene_concepts.json
```

实验和门控结果见 [SCENE_CONCEPT_EXPERIMENT_2026-10-23.md](docs/SCENE_CONCEPT_EXPERIMENT_2026-10-23.md)。

配置修改后重启后端。

## 本地 RAG

已实现字符 n-gram TF-IDF 检索，无需额外模型或向量数据库。知识库包含 15 条规则、4 条核验案例、1 条教学案例。
固定工作流在 OCR 取得文本后查询相关规则；最终处置继续使用程序策略。
55 条查询用于阈值校准，不代表独立测试或真实图片审核准确率。

详细说明见 [RAG 与前端实现说明](docs/RAG与前端实现说明.md)。

## 硬探针与 binary injury

`pic/pic/` 中的 90 张图片用于冻结硬探针评估，不参与训练。当前流程在原有四分类 injury 之外增加一个冻结视觉塔的 binary injury head：

- binary head 负责“是否存在血腥”的校准判断；
- 四分类 injury 继续负责轻微、中度、严重；
- binary 明确阴性时否决血腥维度；若风险 L1/L2 没有独立攻击、威胁或文字支持，则按 `effective_risk` 区分真实冲突与风格假冲突；
- 外部 LLM 识图默认关闭。

训练、评估命令和最新指标见 [HARD_PROBE_2026-10-21.md](docs/HARD_PROBE_2026-10-21.md)。
最终 180 张冻结盲测见 [SOLID_BLIND_2026-10-21.md](docs/SOLID_BLIND_2026-10-21.md)。

## 质量退化检测

质量头已升级为 FG-CLIP2 embedding 与低层清晰度、频域、暗块和块边界特征的融合头：

```text
checkpoints/best/quality_fusion/
```

严重模糊、文字扭曲和故意遮挡作为证据可靠性提示；`X_无法判定` 或内容证据冲突会触发人工复核。180 张质量探针、固定开发/盲测划分和最终指标见
[QUALITY_PROBE_2026-10-22.md](docs/QUALITY_PROBE_2026-10-22.md)。

当前策略版本为 `2026-10-26-effective-v5`：保留 `risk_raw`，由 binary、quality、
场景 Prompt、OCR/RAG 和域证据生成独立的 `effective_risk`。AI 二次元假冲突可主动降级；
真实域 L2、L3、X、近阈值、缺失分数和文字未完成仍需复核。

固定流程、域策略、冻结探针和已知限制见
[EFFECTIVE_POLICY_V5_2026-10-26.md](docs/EFFECTIVE_POLICY_V5_2026-10-26.md)。
动漫风格路由与视觉触发见
[ANIME_VISION_STRATEGY_2026-10-27.md](docs/ANIME_VISION_STRATEGY_2026-10-27.md)。
架构见 [architecture.md](docs/architecture.md)。

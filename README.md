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

- `config/llm.yaml`：DeepSeek，用于条件式解释、文字语义复核和受限的结构化视觉证据复核。
- `config/ocr.yaml`：默认使用本地 GLM-OCR，通过 `http://127.0.0.1:8080/v1/chat/completions`
  整图识别。阿里云和旧 GLM 云接口仅保留用于对照，不会自动回退。

也可在启动终端设置 `LLM_API_KEY`。云端 OCR 仅做对照时才会使用
`ALIYUN_OCR_APPCODE`（旧 GLM Provider 使用 `OCR_API_KEY`）。实际密钥文件已被 Git 忽略。

## 4. 启动后端

```powershell
.\.venv\Scripts\python.exe run_backend.py
```

默认端口 `8000`，健康检查地址：<http://127.0.0.1:8000/health>。返回 `service_ready: true`、`model_loaded: true`、`mock: false` 表示真实模型已就绪。使用 Ctrl+C 停止后端。

## 5. 打开前端

启动后端后，浏览器打开 <http://127.0.0.1:8000/> 即可使用审核工作台。
结果页顶部展示程序动作、`effective_risk`、原始风险、图片域和总耗时；下方按 CLIP 基础头、
专项模型与语境、文字与规则、分支与最终裁决四组展示工具证据，详细分数和原始 JSON 可折叠查看。

左侧审核队列按工作区隔离并使用 SQLite 持久化，分为自动审查区与已完成区；
已完成区包含通过队列和调整队列，支持按域、动作、风险筛选。
人工可确认系统建议或按降级/升级方向多次调整最终动作，并导出当前工作区 CSV。
“导入文件夹”会按文件名排序后串行审核。说明见
[HUMAN_WORKFLOW_2026-10-28.md](docs/HUMAN_WORKFLOW_2026-10-28.md)。

也可以独立启动前端：

另开一个终端，在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

浏览器打开 <http://127.0.0.1:5500/>，或自行使用 Live Server 等 HTTP 服务启动 `frontend/`。不要直接双击 HTML，`file://` 仅提供界面预览。前端后端地址在 `frontend/config.js` 中设置，默认 `http://127.0.0.1:8000/`；后端允许本机 localhost/127.0.0.1 的 HTTP 前端跨端口访问。

更换后端端口时，同时修改 `backend/config.yaml` 和 `frontend/config.js`。
工作区编号与名称保存在浏览器 localStorage，页面时间线按当前标签页会话保留；
审核队列与人工结果保存在 `data/reviews.sqlite3`，审计原图保存在 `backend/logs/images/`。

## 可调整的运行配置

- `config/inference.yaml`：已标定的暴力类型、图像质量阈值。
- `backend/config.yaml` → `features.llm_explanation_mode`：默认 `selective`。放行和限流使用程序固定模板；人工复核、下架才调用 LLM，并按固定 JSON 字段填写解释。
- `backend/config.yaml` → `text_detection`：轻量文字检测参数；默认 `opencv_light`，无需模型权重。
- `backend/config.yaml` → `anime_vision_review`：高置信动漫或“武器类型 + 游戏 UI Prompt”窄候选可调用视觉 LLM。真实图片语义门控代码保留，但 `real_context_enabled=false`，默认不自动改变真实图片动作。
- `audit.save_images`：是否保存审核原图。日志与图片不上传 GitHub。

阿里云 OCR 配置见 [ALIYUN_OCR.md](docs/ALIYUN_OCR.md)。AppCode 只放在本地
`config/ocr.yaml` 或 `ALIYUN_OCR_APPCODE` 环境变量中。

本地 GLM-OCR 的独立环境、模型路径、启动命令和速度对比见
[LOCAL_GLM_OCR.md](docs/LOCAL_GLM_OCR.md)。

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
- 真实图片的外部 LLM 语义放行默认关闭；
- 动漫/游戏 UI 的结构化视觉复核按窄路由启用。

训练、评估命令和最新指标见 [HARD_PROBE_2026-10-21.md](docs/HARD_PROBE_2026-10-21.md)。
最终 180 张冻结盲测见 [SOLID_BLIND_2026-10-21.md](docs/SOLID_BLIND_2026-10-21.md)。

## 质量退化检测

质量头已升级为 FG-CLIP2 embedding 与低层清晰度、频域、暗块和块边界特征的融合头：

```text
checkpoints/best/quality_fusion/
```

严重模糊、文字扭曲和故意遮挡作为证据可靠性提示；`X_无法判定` 或内容证据冲突会触发人工复核。180 张质量探针、固定开发/盲测划分和最终指标见
[QUALITY_PROBE_2026-10-22.md](docs/QUALITY_PROBE_2026-10-22.md)。

当前策略接口版本标识为 `2026-10-26-effective-v5`，包含 2026-10-30 policy v2 修订：

- 保留 `risk_raw`，由 binary、quality、场景 Prompt、OCR/RAG 和域证据生成独立的 `effective_risk`；
- `risk_without_support` 默认改为保守限流；
- `risk_X` 只在类型头、binary、质量、Prompt 与文字证据全部一致阴性时放行；
- OCR 明确食物/酱汁语义可解除红色 L3 假冲突；
- OCR 工具失败本身不再强制转人工；
- 真实图片语义自动放行默认关闭；新闻执法、军警执勤和合法武器展示仅保留为可选实验路由；
- 体育 Prompt 已移除，医学和体育不会触发真实图片视觉语义路由。

2026-10-30 最终 447 张测试结果：444 张成功，合并人工标签更正后
L0 自动误杀 11.46%，L1 自动放行 3.26%，L2–L6 动作放行 1.01%，
总转人工 4.05%。
详细汇总见项目根目录的 `WORK_SUMMARY_2026-10-29.md`。

固定流程、域策略、冻结探针和已知限制见
[EFFECTIVE_POLICY_V5_2026-10-26.md](docs/EFFECTIVE_POLICY_V5_2026-10-26.md)。
语义视觉路由与触发见
[ANIME_VISION_STRATEGY_2026-10-27.md](docs/ANIME_VISION_STRATEGY_2026-10-27.md)。
架构见 [architecture.md](docs/architecture.md)。

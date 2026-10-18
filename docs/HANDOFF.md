# 项目交接摘要

## 当前路线

项目采用 FG-CLIP2 + LLM ReAct Agent。CLIP、OCR、RAG 和低层像素分析都是 Agent Tools：

- `analyze_image` 必须成功调用，调用守卫拒绝无 CLIP 证据的最终提交；
- GLM-OCR、低层特征是可选工具，RAG 当前硬关闭；
- LLM 不直接看图；
- API 只返回 `id` 和 `suggestion`。

详细架构见 `architecture.md`，接口见 `api.md`。

## 标签契约

训练、推理、评估不得偏离当前 Label Studio 四字段：

- risk：5 类单标签；
- violence type：4 路多标签；
- injury：4 类单标签；
- quality：5 类多标签（multi-hot，可同时命中"严重模糊 + 文字扭曲形变"）。

`target` 暂不训练。缺失 type/quality 分别使用 `type_valid`/`quality_valid` 掩码；risk/injury 缺失使用 `IGNORE_INDEX`。checkpoint 的 `labels.json` 带 `schema_version: 2` 与 `task_types`，旧单标签 quality 权重会被拒绝加载。

## 数据现状

数据集构建入口为 `python -m src.build_dataset`，输出 `data/clean/`（labels.json + images/ + report.md/report.json + removed_samples.jsonl + review_extras.json）。

2026-10-11 导出（3045 条任务，全部有图）：清洗后保留 **2922** 条，删除 123 条 = 确定性矛盾 114 条（L0 却标直接暴力/血腥/暗示威胁 94 条、违规等级却无暴力且无血腥 5 条、违规等级却缺失表现字段 12 条、无暴力+其他类型 3 条）+ 标注全部取消 9 条，通过率 **95.96%**。原始图片目录里 500 张没有任何标签引用的图片已删除，未进入数据集。

`L0_安全` 却严重模糊/文字扭曲（471 条）与 `X_无法判定` 却标「清晰」只进复核清单，不自动改标签。

## 当前训练结果（2026-10-18）

第一版权重已完成训练：`checkpoints/best/`（LoRA adapter + head.pt + labels.json schema v2 + 验证集指标）。最多 20 epoch，第 4 轮最佳、第 9 轮早停，耗时约 14 分钟。

测试集（147 条）：risk macro-F1 0.6215 / L3 recall 0.8571；type micro-F1 0.8224 / macro-F1 0.7757；injury accuracy 0.8116 / macro-F1 0.4614；quality micro-F1 0.5796 / macro-F1 0.4556（文字扭曲 recall 0.929）。多标签阈值已在验证集标定并写入 `config/inference.yaml`。

主要短板：故意遮挡全量仅 23 条（验证/测试各 3 条），基本不可用；文字扭曲精确率低（0.181），但按项目目标选择了召回优先。数据翻倍后需重新标定阈值。详见 `docs/TRAINING_REPORT_2026-10-18.md`。

## 已完成

- 修复 ReviewEngine 的 `lora` 作用域错误；
- 修复 `WeightedCE/focal_gamma` 和 `nn.Module.type` 命名冲突；
- 接通 FG-CLIP2 processor、视觉塔和四任务头推理；
- 统一 head 与视觉塔的 device/dtype；
- 接通多标签缺失掩码；
- 实现 LoRA adapter + head 的最小 checkpoint；
- 实现受控 ReAct 图、CLIP 调用守卫和请求缓存；
- 实现 Mock Engine/LLM/OCR；
- 固定并完成智谱 GLM-OCR 的 JSON+Base64 接口适配；
- 固定并完成 DeepSeek-V4-Flash 独立配置与 LangChain 适配；
- 增加工具级 RAG 禁用开关；
- 建立项目内 Python 3.12 + CUDA 12.8 环境并验证 RTX 5060 Ti；
- 使用 `knife_man.jpg` 完成 FG-CLIP2 原始权重真实前向；
- 使用 DeepSeek-V4-Flash 完成真实 Agent 全流程，轨迹为 `analyze_image → analyze_lowlevel → finalize_review`；
- 修正 GLM-OCR Base64 data URI 并完成真实接口测试；
- 实现数据配对、标签头收敛与确定性清洗（`src/build_dataset.py`），并接入 `data/clean/`；
- 让 `config/inference.yaml` 真正被 `src/cli.py` 读取，补齐 quality 多标签阈值；
- 自动化测试 48 项（含 9 项不依赖 torch 的数据与配置测试）；
- 冻结前端接口仍为正常/降级 `{id, suggestion}`、HTTP 错误 `{code, message, id}`；
- 更新 `.gitignore`，排除整个模型目录、权重、标注导出、数据、密钥和运行产物；
- 修复 API 图片校验和错误结构；
- 增加最小训练、评估入口与自动化测试；
- 更新 README、架构与 API 文档。

## 外部条件与后续工作

1. 完成全部标注并导出最终 JSON；
2. 提供完整本地图片目录；
3. 在训练机上准备环境（当前 WSL Ubuntu 的 conda 环境 `fgclip`：`/home/ygx/miniconda3/envs/fgclip`，Python 3.10.21 + torch 2.5.1+cu124，RTX 4060 Laptop 8GB，CUDA 可用；只缺 `peft`）；
4. 运行最终数据体检，复核语义张力记录；
5. 生成并冻结 `data/split.json`；
6. 正式训练 LoRA 和四任务头；
7. 在验证集标定 violence type 阈值；
8. 在独立测试集生成正式指标；
9. 为前端联调补充跨域代理或明确部署为同源；
10. 如结题需要，再实现 RAG 判例库；
11. 正式上线前增加请求级总超时和明确的 504 语义。

## 运行

WSL 训练/联调环境（Windows 侧代码位于 `D:\Image-Review-Agent`，WSL 内为 `/mnt/d/Image-Review-Agent`）：

```bash
wsl -d Ubuntu
conda activate fgclip
cd /mnt/d/Image-Review-Agent
python -B -m unittest discover -s tests -v      # 全量测试
python -m src.train --smoke-steps 1             # 冒烟训练
```

后端联调额外需要 `config/llm.yaml`、`config/ocr.yaml` 与 API Key：

```bash
export LLM_API_KEY="你的 DeepSeek API Key"
export OCR_API_KEY="你的智谱 API Key"
python run_backend.py
```

当前 `require_checkpoint=false`，允许随机任务头验证链路，但结果不能作为模型效果。正式训练完成后必须改回 `lora=true`、`require_checkpoint=true` 并指定 checkpoint。

## API 联调口径

- 正常和业务降级均返回 HTTP 200 `{id, suggestion}`；当前没有 `degraded` 字段；
- 图片/字段错误返回 HTTP 400 `{code, message, id}`；服务未初始化返回 503；
- 当前没有请求级总超时和 504，前端超时暂建议 120 秒；
- `/health.mock=false` 仅代表真实视觉引擎，不代表分类头已经训练；
- RAG 当前未注册给 Agent；
- 完整细节以 `docs/api.md` 为准。

## Git 协作口径

仓库只提交代码、测试、文档和无密钥配置模板。`fg_clip/`、checkpoint、数据、阶段性标注 JSON、测试图片、真实密钥配置、虚拟环境和日志全部忽略。任何真实密钥一旦进入 Git 历史必须立即轮换，不能只靠后续删除文件解决。

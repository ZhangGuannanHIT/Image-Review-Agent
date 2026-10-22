# 项目交接摘要

更新时间：2026-10-23

## 1. 当前路线

项目采用 `FG-CLIP2 + LoRA + 分类头 + 受控 ReAct Agent`：

- `analyze_image` 必须成功调用，调用守卫拒绝无 CLIP 证据的最终提交；
- 四任务头继续输出 risk、violence type、injury 和 quality；
- 独立 binary injury 头负责校准后的“是否存在血腥”判断；
- quality fusion 头融合 CLIP embedding 与 15 维低层质量特征；
- OCR、低层分析、RAG 和场景概念是受控工具；
- 程序先完成 `allow / limit / review / block`，LLM 最后只生成解释；
- LLM 不直接接收原图，也不能改变程序处置；
- API 稳定返回 `{id, suggestion}`。

详细架构见 `architecture.md`，接口见 `api.md`。

## 2. 标签与 checkpoint

训练和推理使用 Label Studio 四任务标签：

- risk：5 类单标签；
- violence type：4 路多标签；
- injury：4 类单标签；
- quality：5 类多标签。

主 checkpoint 位于 `checkpoints/best/`，包含 LoRA adapter、四任务头和标签定义。
独立部署头包括：

- `checkpoints/best/injury_binary/`：血腥存在性；
- `checkpoints/best/quality_fusion/`：视觉 embedding + 低层特征融合质量头。

真实模式设置 `require_checkpoint=true`，缺少任一必要权重时服务不会 ready。

## 3. 数据与冻结探针

- 主训练集：2922 条清洗后的标注图片；
- `pic/`：90 张冻结硬探针，不参与训练；
- `solid/`：180 张血腥程度盲测，AI 严重血腥格为空；
- `D:\quality`：180 张质量退化探针，固定划分为 60 张开发集和 120 张最终盲测。

当前可报告的关键结果：

- binary injury 在 solid 上灵敏度 99.0%，无血假阳性率 0.0%，AUROC 0.999875；
- v4 下 solid 有血图没有自动放行，无血图没有自动限流或下架；
- solid 无血图为 54 张放行、26 张人工复核；
- quality 最终盲测 macro-F1 0.572；严重模糊 0.817、文字扭曲 0.600、遮挡 0.456；
- v4 下 quality 安全图 60 张全部未自动限流/下架；
- v4 下 quality 违规图仅 3 张放行、18 张复核、38 张限流、1 张下架；
- v4 的人工复核量高于 v3，这是从“不确定放行”切换为“不确定复核”的直接结果；
- OCR 小样本证明文字扭曲可以用 OCR/RAG 补充证据，但样本量不足以报告正式泛化指标。

详细报告：

- `docs/HARD_PROBE_2026-10-21.md`
- `docs/SOLID_BLIND_2026-10-21.md`
- `docs/QUALITY_PROBE_2026-10-22.md`
- `docs/OCR_TEXT_DISTORTION_PILOT_2026-10-22.md`
- `docs/SCENE_CONCEPT_EXPERIMENT_2026-10-23.md`
- `docs/POLICY_V4_REGRESSION_2026-10-23.md`

## 4. 当前处置策略

策略版本：`2026-10-23-v4`。

- 高置信、无冲突的 L0 可以放行；
- 风险 `X_无法判定`、分数缺失、风险与内容头冲突、证据不足、
  四分类血腥与 binary injury 冲突时进入人工复核；
- 风险头支持轻中度违规，并有独立类型、血腥或 OCR 高风险证据时限流；
- 风险头支持严重违规，并有严重血腥或攻击进行中证据时下架；
- 用户确认的业务口径是“高分不确定性进入人工复核”，不再采用不确定放行；
- 质量退化只作为可靠性警告，不单独改变动作。

此前针对 binary injury、quality fusion、OCR、场景概念和 L0 条件短路的工作，
目标是保持违规召回的同时减少无血图误杀和不必要人工复核。solid 上效果明确；
quality 集仍是下一阶段回归验证和限制说明的重点。

## 5. OCR、RAG 与场景概念

OCR 当前使用阿里云高精版接口：

- 配置：`config/ocr.yaml`；
- 鉴权：`Authorization: APPCODE <AppCode>`；
- 只有 quality 命中文字扭曲、低层检测疑似文字或文字探测失败时才自动 OCR；
- OCR 结构化状态为 `empty / partial / readable / failed`；
- 日志不保存 OCR 原文。

本地 RAG 使用字符 n-gram TF-IDF，包含规则、核验案例和教学案例。
Agent 的通用 `search_knowledge` 结果会去重后写入 `decision_evidence`。

场景概念是独立旁路：

- 非 L0 时由 `analyze_image` 自动执行一次；
- 常规场景只补充解释；
- 新闻、医学等安全语境必须有场景 Prompt 与 OCR/RAG 第二证据共同确认；
- 确认后视觉违规最多降为限流，绝不直接放行；
- 新闻语境与未加工严重血腥暴力冲突时进入人工复核。

## 6. 完整证据与最终解释

`build_decision_evidence()` 汇总：

- risk / type / injury / quality；
- binary injury；
- 低层特征和文字探测状态；
- OCR 状态与规则命中；
- 场景概念结果；
- Agent 通用 RAG 命中。

程序 `decide()` 锁定动作后，独立解释 LLM 接收完整 `decision_evidence` 和只读
`program_decision`。解释不得携带工具调用或自行输出动作词；校验失败时回退到
`render(decision)`。日志通过 `final_explanation.status` 和 `suggestion_source`
区分 LLM 解释与确定性回退。

## 7. 当前测试状态

最近一次全量单元和契约测试：138 项。

已覆盖：

- 策略动作与不再“不确定放行”；
- binary injury 与原 injury 的冲突处理；
- quality fusion 与阈值；
- OCR、RAG、场景概念和低层证据；
- L0 条件短路及文字探测失败降级；
- 完整证据包；
- LLM 最终解释的动作锁和回退；
- API 契约、图片校验与日志脱敏。

单元测试不能替代业务回归。提交 GitHub 前仍需运行 solid、quality、OCR 小样本
和真实后端端到端验证。

## 8. 运行

WSL 训练/联调环境：

```bash
wsl -d Ubuntu
source /home/ygx/miniconda3/etc/profile.d/conda.sh
conda activate fgclip
cd /mnt/d/Image-Review-Agent
python -B -m unittest discover -s tests
```

真实后端需要本地 `config/llm.yaml` 和 `config/ocr.yaml`：

```bash
export LLM_API_KEY="你的 DeepSeek API Key"
export ALIYUN_OCR_APPCODE="你的阿里云 AppCode"
python run_backend.py
```

流程演示可设置 `BACKEND_MODE=mock`。Mock 只用于界面和契约联调，不产生正式指标。

## 9. GitHub 与安全口径

- 真实密钥、图片、基础模型、大部分实验产物和运行日志均已忽略；
- `config/llm.yaml`、`config/ocr.yaml` 只保留本地，不提交；
- 仓库只提交代码、测试、文档、无密钥配置模板和明确放行的小型部署 checkpoint；
- 一旦密钥曾进入 Git 历史，必须轮换密钥，不能只删除当前文件；
- 曾暴露过的阿里云 AppSecret 建议在正式提交前轮换；
- 请求级总超时和明确 504 语义仍未实现，属于正式上线前工作。

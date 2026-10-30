# 项目交接摘要

## 当前架构

`FG-CLIP2 + fixed conditional workflow + effective risk v5 policy v2 + constrained visual LLM`

当前本机 WSL2 Ubuntu 虚拟磁盘位于 `D:\WSL\Ubuntu\ext4.vhdx`；
默认用户仍为 `ygx`，`fgclip` 与 `glmocr` conda 环境随发行版保留。

主入口：

- [ReviewService](../backend/service.py)
- [FixedReviewWorkflow](../backend/workflow.py)
- [Effective risk policy](../backend/policy.py)
- [Lightweight text detection](../backend/text_detection.py)

已删除：

- LangGraph ReAct StateGraph；
- LLM 工具选择；
- `finalize_review` 和调用守卫；
- 旧 `agent/state/prompts/parser` 模块。

## 请求流程

1. API 校验图片。
2. FG-CLIP2 输出 risk/type/injury/quality fusion。
3. binary injury 输出血腥存在性。
4. 可选血腥视觉复核。
5. 每张图执行轻量文字检测。
6. 检测命中或失败时触发 OCR。
7. OCR 有文本时执行本地 RAG。
8. risk 非 L0 时执行场景 Prompt。
9. `effective_risk` 策略生成 allow/limit/review/block。
10. 动漫或游戏 UI 窄路由进入结构化视觉复核。
11. 真实图片语义路由代码保留，但 `real_context_enabled=false`。
12. 解释 LLM 只生成解释；动漫/游戏 UI 使用独立的结构化视觉 LLM 证据，程序仍锁定最终动作。
13. 写入脱敏日志，返回 `{id, suggestion}`。

## 策略

- 原始 `risk_raw` 永不删除。
- binary 判断为 positive、uncertain、clear negative 三段。
- binary 阴性只否决血腥支持，不能删除独立攻击/威胁/文字证据。
- AI 二次元域 L1/L2 假冲突可主动降级。
- 动漫视觉只有在 injury/attack/threat 全 absent 时才可撤回；不确定或 mild 保持 L1。
- `risk_without_support` 无法解除时改为保守限流。
- `risk_X` 只在类型头、binary、质量、场景 Prompt 和 OCR/RAG 全部一致阴性时放行。
- OCR 明确食物/酱汁语义可解除红色 L3 假冲突，但不能单独决定动作。
- OCR 工具失败本身不再强制转人工。
- 真实域 L2、L3、X、近阈值、缺失分数保留保守处置。
- 安全语境最多把下架降为限流。
- 高风险 OCR 文字可独立把 L0 提升为限流。

真实图片语义实验：

- 仅保留 `news_legal_context`、`authority_context`、`harmless_weapon`；
- 体育 Prompt 已移除，医学不触发真实视觉；
- 默认关闭；
- 即使启用，直接暴力类型分数 `>=0.65` 也不允许视觉撤回。

详细口径见 [EFFECTIVE_POLICY_V5_2026-10-26.md](EFFECTIVE_POLICY_V5_2026-10-26.md)。

## 当前冻结探针

solid visual v5：

- 180 张：放行 72、复核 9、限流 90、下架 9；
- 真实 100：放行 38、复核 2、限流 51、下架 9；
- AI 80：放行 34、复核 7、限流 39；
- 违规自动放行 0。

quality visual v5：

- 120 张：放行 50、复核 30、限流 39、下架 1；
- AI 违规自动放行 0；
- 真实违规自动放行 3，与 v4 是同一组文件，没有新增。

以上两轮没有调用外部 OCR。生产文字链路的复核量需在配置 OCR 后重跑。

## policy v2 最终端到端

2026-10-30，正式 API 顺序执行 447 张，444 张成功，3 张 AVIF 被拒绝：

| 真值组 | N | allow | limit | block | review |
|---|---:|---:|---:|---:|---:|
| L0 | 253 | 211 | 29 | 0 | 13 |
| L1 | 92 | 3 | 84 | 1 | 4 |
| L2–L6 | 99 | 1 | 82 | 15 | 1 |

- 上一轮 15 张 L1 放行中确认只有 `BT0186`、`BT0332` 属实；其余 13 张已在主 `manifest.csv` 内直接改为 L0。
- 总转人工：18/444，4.05%；
- 本地 OCR：399 次，397 成功、2 失败；
- 语义视觉：159 次，159 成功；
- 真实图片语义自动放行未启用；
- L2–L6 的 1 张 allow 是视觉漂移样本 `40`，两次视觉结果不一致。

结果位于 `eval_results/final_e2e_20261030_policy_v2/`。

## 文字检测

默认 `LightweightTextDetector`：

- OpenCV/CPU，无新模型；
- 高召回，误触发只增加 OCR 调用；
- 检测失败按需要 OCR 处理；
- `quality=文字扭曲形变` 不再单独触发 OCR。

当前 OpenCV 方案选择性有限，solid 扫描触发率约 87.8%。有本地 RapidOCR 时可切换：

```yaml
text_detection:
  provider: "rapidocr"
```

## 日志

OCR 原文不写日志。保留状态、字符数、SHA-256、规则 ID、区域数、耗时和结构化决策。

## 测试

WSL conda `fgclip`：

```bash
source /home/ygx/miniconda3/etc/profile.d/conda.sh
conda activate fgclip
cd /mnt/d/Image-Review-Agent
python -B -X utf8 -m unittest discover -s tests -v
```

当前 215 项通过。

## 待办

1. 人工核验视觉漂移样本 `40`，并评估多采样一致性或强模型头冲突保护。
2. 核验 16 张 L1 自动放行，优先复查 `BT0119`、`BT0322`。
3. 核验 29 张 L0 自动限流，重点是真实域军警、持械、医学和新闻场景。
4. 为真实图片研究独立的语义分类器和世界关系推理，不使用单次图文相似度直接放行。
5. 完成 159 份结构化视觉输出的字段级人工核验。
6. 修复 AVIF 输入兼容或在前端明确拒绝。
7. 处理本地 OCR 两次失败和 P95 约 19.4 秒的长尾。
8. 扩大动漫违规正样本，覆盖武器指向、直接攻击、文字威胁和虐待。

## Git 边界

仓库只提交代码、测试、文档和无密钥配置模板。数据、图片、日志、真实 API Key、基础模型和大权重不提交。

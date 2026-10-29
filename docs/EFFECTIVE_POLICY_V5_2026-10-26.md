# 固定证据流程与 effective risk v5 policy v2（2026-10-30）

> 2026-10-30 policy v2 更新：真实图片语义自动放行仍默认关闭；本文件补充 OCR
> 图片语义、红色食物联合规则、risk X 多证据放行、保守限流和
> `risk_without_support` 的新处置。

## 1. 变更目标

- 删除 LangGraph/ReAct 工具选择循环。
- 保留 CLIP risk 原始输出，新增策略级 `effective_risk`。
- 将“真实冲突”和“风格/域导致的假冲突”分开处置。
- 文字检测先于 OCR；quality 的“文字扭曲形变”不再单独触发 OCR。
- OCR、RAG、场景 Prompt、binary 和 quality 只作为可审计证据，不覆写原始模型输出。
- 真实图片语义自动放行代码保留，但 `real_context_enabled=false`，默认不改变动作。
- 当前真实图片只把场景 Prompt、OCR 描述和合法武器/新闻执法候选作为解释和后续研究信号。

## 2. 固定流程

```text
读取图片
  → FG-CLIP2 risk/type/injury/quality fusion
  → binary injury
  → 条件性血腥视觉复核（默认关闭）
  → 轻量文字区域检测
  → 检测命中或失败时调用 OCR
  → OCR 有文本时检索 RAG 并执行逐字 quote 语义复核
  → risk 非 L0 时匹配场景 Prompt
  → effective risk 策略
  → 动漫/游戏 UI 条件视觉复核
  → 真实图片语义路由仅作可选实验，默认关闭
  → allow / limit / review / block
  → 无工具 LLM 生成解释
```

所有请求都会执行文字检测，包括高置信 L0。只有文字检测明确为空时才跳过 OCR。

## 3. effective risk 口径

`risk_raw` 始终保留 risk 头原始 argmax。策略另行输出 `effective_risk`，并记录 `conflict_type` 和 `adjustments`。

### binary

- `p_blood >= threshold`：positive。
- `p_blood <= threshold - binary_clear_margin`：clear negative。
- 中间区域：uncertain，不进入激进降级。

clear negative 只否决血腥维度。若存在直接攻击或明确威胁，类型证据仍可支持限流或下架。

### 假冲突

同时满足以下条件时视为风格/质量型假冲突：

- binary clear negative；
- risk 为 L1/L2；
- 无直接暴力、武器威胁或高风险文字；
- 无达到门槛的独立内容支持。

在此基础上：

- 二次元/插画域 Prompt 命中：可降为有效 L0。
- 动漫域 quality 异常：可辅助解释风险头不稳定候选。
- 动漫域 L3 不再仅因原风险等级或仅持械而保留 L1。
- 视觉 injury/attack/threat 三个关系字段均为 absent，且没有 uncertain、horror 或 mild 时，
  可以撤回原限流；任一关系不确定仍保持 L1。
- 场景 Prompt 命中红色物体/颜料等混淆项：动漫域可降级；真实域仅低置信 L1 可用。
- 真实域 L1 且低置信、无严重模糊/遮挡、无文字候选：可窄口径放行。
- 真实域 L2：缺少独立支持时改为保守限流，不再停在 `risk_without_support` 复核。
- L3：不能仅凭 binary 阴性或域标签自动降级；OCR 明确食物/酱汁语义、红色液体 Prompt、
  binary 阴性和无独立暴力同时成立时，使用窄规则放行。
- X：仅当类型头只有“无暴力”、binary 明确阴性、质量不是严重模糊/故意遮挡、
  常规违规 Prompt 风险 margin 不高于 0.015、OCR/RAG 未发现威胁时放行；其余继续复核。
- 分数缺失、近阈值、OCR 检测到文字但未完成：复核。
- OCR 工具失败本身不再强制转人工，失败事件保留在工具状态和日志中。

### 安全语境

- 场景 Prompt 单独命中不是豁免。
- 场景安全语境必须有 OCR/RAG 第二证据。
- 已确认安全语境最多把下架降为限流，不能直接放行。
- 新闻语境与严重未加工暴力同时出现时仍视为新闻壳风险并复核。

### OCR 图片描述

本地 GLM-OCR 仍可能出现“图中没有文字、但模型描述了图片内容”的现象。当前策略：

- 轻量文字检测高召回，可能把纹理或界面误当成文字区域；
- OCR 返回自由文本后，新增 `content_profile` 区分普通文字和图片描述；
- `safe_food_scene` 和 `safe_red_liquid` 只在明确食物、酱汁、番茄等词存在，
  且没有 blood/wound/attack/weapon 等伤害词时为真；
- OCR 描述不能单独修改动作，也不作为高置信视觉真值；
- 红色食物规则还要求 L3、binary 明确阴性、场景红色液体标签、无独立暴力和无文字威胁。

### 真实图片语义门控（默认关闭）

实验代码支持：

- `news_legal_context`；
- `authority_context`；
- `harmless_weapon`。

医学和体育不触发真实图片视觉语义路由。即使显式启用，还需：

- binary 明确阴性；
- 直接暴力类型分数低于 0.65；
- 视觉返回 `real_photo`；
- injury/attack/threat 全部 absent。

2026-10-30 最终测试未启用该能力，真实图片仍按原策略保守处理。

### policy v2 端到端结果

444 张成功样本中，合并 13 张人工标签更正后：

- L0：253 张，放行 211、限流 29、转人工 13；
- L1：92 张，放行 3、限流 84、下架 1、转人工 4；
- L2–L6：放行 1、限流 82、下架 15、转人工 1。

该 1 张 L2–L6 动作放行是视觉输出漂移样本 `40`，不作为稳定规则能力。

## 4. 冻结探针结果

以下为 visual-policy 消融结果，不调用外部 OCR。solid/quality 均使用当前真实 checkpoint、binary、quality fusion 和已保存的场景 Prompt 分数。

### solid 180

| 版本 | 放行 | 复核 | 限流 | 下架 | 违规自动放行 |
|---|---:|---:|---:|---:|---:|
| v4 | 54 | 27 | 90 | 9 | 0 |
| v5 visual | 72 | 9 | 90 | 9 | 0 |

分域：

| 域 | 放行 | 复核 | 限流 | 下架 | 违规自动放行 |
|---|---:|---:|---:|---:|---:|
| real 100 | 38 | 2 | 51 | 9 | 0 |
| AI 80 | 34 | 7 | 39 | 0 | 0 |

### quality 120

| 版本 | 放行 | 复核 | 限流 | 下架 | 违规自动放行 |
|---|---:|---:|---:|---:|---:|
| v4 | 49 | 32 | 38 | 1 | 3 |
| v5 visual | 50 | 30 | 39 | 1 | 3 |

分域：

| 域 | 放行 | 复核 | 限流 | 下架 | 违规自动放行 |
|---|---:|---:|---:|---:|---:|
| real 60 | 25 | 19 | 15 | 1 | 3 |
| AI 60 | 25 | 11 | 24 | 0 | 0 |

v5 的 3 张真实违规自动放行与 v4 是同一组文件，没有新增。它们属于模型 risk 头将违规图判为 L0 的基线自动误放，不能通过 binary/quality 事后规则可靠修复。

## 5. 文字检测门控

默认 `opencv_light`：

- 全流程 CPU，无模型下载；
- 目标是高召回，误触发只增加 OCR，不会直接改变动作；
- 检测失败按“需要 OCR”处理。

已有 `quality_ocr_probe` 20 张小样本：

- OCR 非空样本检测召回：9/10；
- OCR 为空但检测触发：8/10。

该探针是文字/扭曲专项集，OCR 为空不等于图中绝对无文字，因此不能把 80% 当作真实无文字图片的 FPR。solid 扫描触发率为 87.8%，说明 OpenCV 方案的选择性有限。生产环境若有本地 RapidOCR，可将 `text_detection.provider` 改为 `rapidocr`。

## 6. 日志

日志不保存 OCR 原文。保存内容为：

- OCR 成功/失败、字符数、平均置信度和请求 ID；
- 规范化文本 SHA-256；
- RAG 规则 ID；
- 文本检测状态、区域数量和置信度；
- 工具步骤和结构化决策证据。

## 7. 复现

```bash
python -B -m unittest discover -s tests -v

python -B -m src.evaluate_effective_policy \
  --kind solid \
  --manifest data/hard_probe_solid/manifest.csv \
  --image-root solid \
  --scene-probe eval_results/scene_concepts_solid.json \
  --ignore-text \
  --out eval_results/solid_effective_v5_visual.json

python -B -m src.evaluate_effective_policy \
  --kind quality \
  --manifest data/quality_probe/final.csv \
  --image-root ../quality \
  --scene-probe eval_results/scene_concepts_quality_final.json \
  --ignore-text \
  --out eval_results/quality_final_effective_v5_visual.json
```

## 8. 仍需补测

- 独立安全二次元集：建议 150 至 300 张，IID 之外另留 test。
- 违规二次元正集：先使用已有 solid AI 40 张有血探针，后续补充武器、文字威胁、虐待等类型。
- 真实域更多 L0 自动误放样本，尤其模糊、局部文字和遮挡。
- OCR 全链路回放，确认文字检测触发后多少图片能从复核转为明确动作。
- 动漫域 Prompt 阈值的独立校准；当前仅用于策略降级，不能当作正式风格分类指标。

# 项目交接摘要

## 当前架构

`FG-CLIP2 + fixed conditional workflow + effective risk + explanation-only LLM`

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
10. LLM 只生成解释；失败时使用确定性模板。
11. 写入脱敏日志，返回 `{id, suggestion}`。

## 策略

- 原始 `risk_raw` 永不删除。
- binary 判断为 positive、uncertain、clear negative 三段。
- binary 阴性只否决血腥支持，不能删除独立攻击/威胁/文字证据。
- AI 二次元域 L1/L2 假冲突可主动降级。
- 真实域 L2、L3、X、近阈值、缺失分数和 OCR 未完成保留复核。
- 安全语境最多把下架降为限流。
- 高风险 OCR 文字可独立把 L0 提升为限流。

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

当前 177 项通过。

## 待办

1. 360 张二次元安全候选已完成初筛和固定划分，人工复核估计约 0.5-1% 为 L1，
   已有 126 张人工确认真值（124 L0、2 L1），其余 234 张未确认，见
   [ANIME_SAFE_SCREEN_2026-10-27.md](ANIME_SAFE_SCREEN_2026-10-27.md)。
   数据暂不能作为零误差 hard negatives，当前不从这些样本直接放宽生产策略。
   120 张独立标注批次和 Label Studio 配置位于
   `D:\动漫图片_待标注_2026-10-27`。
   纯策略达到 10 限流/5 复核的上限分析见
   [ANIME_POLICY_CEILING_2026-10-27.md](ANIME_POLICY_CEILING_2026-10-27.md)。
2. 补充违规二次元武器、文字威胁、虐待等类型。
3. 用真实 OCR 回放 solid/quality，量化文字检测门控后的复核变化。
4. 分析 quality 中 3 张 v4 就存在的真实 L0 自动放行。
5. 若延迟不可接受，接入并验证 RapidOCR 检测后端。
6. 训练动漫域校准层，优先修正 binary/risk 在二次元域的假阳性。
7. 动漫风格路由与 LLM 视觉复核已接入，见
   [ANIME_VISION_STRATEGY_2026-10-27.md](ANIME_VISION_STRATEGY_2026-10-27.md)。
   视觉触发仍约 171/360，需在真实业务量下评估 API 成本和延迟。

## Git 边界

仓库只提交代码、测试、文档和无密钥配置模板。数据、图片、日志、真实 API Key、基础模型和大权重不提交。

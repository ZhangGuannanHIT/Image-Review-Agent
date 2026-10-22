# 策略 v4 冻结盲测回归（2026-10-23）

## 1. 变更

本次将“高分不确定性”的处置从放行改为人工复核：

- 分类分数缺失、模型未训练、风险与内容头冲突、风险 X、
  binary injury 与四分类血腥证据冲突，以及无法形成明确动作的候选统一进入复核；
- 高置信、无冲突的 L0 仍可放行；
- binary injury、quality fusion、OCR、RAG 和场景概念逻辑不变；
- `decision_evidence` 新增完整低层特征及其文字探测状态，供最终 LLM 解释使用。

策略版本：`2026-10-23-v4`。

## 2. Solid 180 张

命令：

```bash
python -B -m src.evaluate_hard_probe_policy \
  --manifest data/hard_probe_solid/manifest.csv \
  --image-root solid \
  --config config/inference.yaml \
  --backend-config backend/config.yaml \
  --out eval_results/solid_policy_v4.json
```

| 真值 | 放行 | 人工复核 | 限流 | 下架 |
|---|---:|---:|---:|---:|
| 无血 80 | 54 | 26 | 0 | 0 |
| 有血 100 | 0 | 1 | 90 | 9 |

与 v3 对比：

- 有血图动作不变，仍为 0 自动放行；
- 无血图自动限流/下架仍为 0；
- 无血图人工复核由 11 张增加到 26 张，放行由 69 张减少到 54 张；
- 增加部分集中在 AI 域，符合“高不确定性进入复核”的新口径。

## 3. Quality 120 张最终盲测

命令：

```bash
python -B -m src.evaluate_quality_probe \
  --manifest data/quality_probe/final.csv \
  --image-root ../quality \
  --config config/inference.yaml \
  --backend-config backend/config.yaml \
  --fusion-head checkpoints/best/quality_fusion \
  --out eval_results/quality_final_v4.json
```

| 真值 | 放行 | 人工复核 | 限流 | 下架 |
|---|---:|---:|---:|---:|
| 安全 60 | 46 | 14 | 0 | 0 |
| 违规 60 | 3 | 18 | 38 | 1 |

与 v3 对比：

- 违规图自动放行由 20 张降到 3 张；
- 违规图非放行覆盖由 40/60 提升到 57/60；
- 安全图仍为 0 张自动限流或下架；
- 安全图人工复核由 12 张增加到 14 张；
- 违规图人工复核由 1 张增加到 18 张；
- 总人工复核由 13 张增加到 32 张。

## 4. 解释

v4 用更多人工复核换取明显更低的违规漏放，同时没有引入新的安全图自动限流或下架。
这说明“高不确定性复核”是有效的安全兜底，但不能宣称已经同时降低人工量。

额外试验过“把 L0 自身置信度低于 0.75 也视为不确定”的方案，结果是：

- solid 无血图变为 8 张放行、72 张人工复核；
- 有血图动作基本不变；
- 人工复核量过高，不能实现项目希望的业务价值。

因此该方案不作为默认策略。最终保留的“高不确定性”指风险 X、证据缺失、
多头冲突或无法形成稳定动作，而不是把多头一致、只是 softmax 分数不高的 L0
直接转人工。L0 快速结束仍使用更严格的 0.75 门槛，但它只影响是否跳过可选工具，
不直接决定最终处置。

当前可以主张：

- binary injury 在 solid 上显著减少了无血误杀；
- v4 在 quality 盲测上把 60 张违规图的自动放行压缩到 3 张；
- 安全图没有被自动限流或下架。

当前不能主张：

- 所有场景都同时实现低误杀、低漏放和低人工量；
- quality 的遮挡和文字扭曲已经彻底解决；
- 3 张违规放行已经是最终状态。

## 5. 后续

1. 对 3 张违规放行图做逐图错误分析，判断是风险头 L0、binary 漏报还是策略分类问题。
2. 对 32 张人工复核做原因分组，区分真正不可判定和可以进一步自动化的分支。
3. 在不增加安全图自动限流/下架的前提下，优先减少安全图复核。
4. 用真实服务跑 OCR、场景概念和最终解释的端到端延迟。

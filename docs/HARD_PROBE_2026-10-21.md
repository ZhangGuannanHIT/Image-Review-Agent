# 硬探针与 binary injury 实验（2026-10-21）

## 1. 目的

验证当前 FG-CLIP2 违规识别模型在真实照片与 AI 二次元图像上的血腥误报，并评估不引入第二个视觉模型的止损方案。

硬探针只用于评估，不参与训练或阈值搜索。图片目录为 `pic/pic/`，已加入 `.gitignore`。

## 2. 数据组成

| 子集 | 有血 | 无血 |
|---|---:|---:|
| AI 二次元高红 | 10 | 10 |
| AI 二次元低红 | 10 | 10 |
| 真实照片高红 | 11 | 8 |
| 真实照片低红 | 14 | 10 |
| AI 写实高红 | 0 | 7 |

总计 90 张，其中有血 45 张、无血 45 张。

限制：

- AI 与二次元高度绑定，不能分离两种因素的独立贡献。
- AI 写实只有 7 张无血图，只能粗略估计假阳性率。
- 当前只有 binary 真值，轻/中/重等级待 Label Studio 补标。

## 3. 数据泄漏检查

对 90 张探针与 `data/clean/images` 做 SHA-256 完全匹配和 64 位 dHash 近重复检查：

- 完全重复：0
- dHash 距离 <= 6：0

## 4. 当前四分类模型基线

阈值 0.75：

| 指标 | 结果 |
|---|---:|
| 总体灵敏度 | 100% |
| 总体假阳性率 | 53.3% |
| 真实照片无血假阳性率 | 6% |
| AI 二次元无血假阳性率 | 100% |

AI 二次元内部 injury AUROC 约 0.97，说明主要是域级分数偏移，而非完全无法排序。

## 5. 诊断结论

- Hue 旋转没有降低高阈值假阳性率。
- 普通 5 裁剪使假阳性率更高，暂不上线。
- FG-CLIP2 的动漫 style prompt margin 区分动漫/真实照片 AUROC 约 0.92。
- prompt 单独区分血腥不足，更适合作为辅助特征。

## 6. Binary injury head

训练方式：

- 冻结现有 LoRA 视觉塔。
- 从四类 injury 派生有血/无血标签。
- 使用 72 张训练集红色无血候选作为 hard negatives，权重为 2。
- 在验证集上选择 `FPR <= 10%` 时灵敏度最高的阈值。

结果：

| 数据集 | 阈值 | 灵敏度 | 无血假阳性率 | AUROC |
|---|---:|---:|---:|---:|
| 验证集 | 0.285 | 91.7% | 8.7% | - |
| 原测试集 | 0.285 | 92.9% | 5.45% | 0.974 |
| AI/二次元硬探针 | 0.285 | 93.3% | 6.7% | 0.987 |

硬探针分片：

| 子集 | 灵敏度 | 假阳性率 |
|---|---:|---:|
| AI 二次元 | 90.0% | 10.0% |
| 真实照片 | 96.0% | 0% |
| AI 写实无血 | - | 14.3%（1/7） |

## 6.1 CORAL ordinal injury 对照

冻结同一视觉 embedding，使用 CORAL 累积损失、类别平衡和 hard-negative 加权训练序数头。

| 数据 | macro-F1 | QWK | 严重度召回 |
|---|---:|---:|---:|
| 原测试集 | 0.480 | 0.833 | 66.7% |
| 90 张硬探针 | 0.579 | 0.799 | 81.0% |

硬探针相邻等级准确率为 87.8%。轻微类只有 3 张，模型未识别出该类，因此轻微等级目前不能下结论。

当前部署仍保留原四分类 head，ordinal head 只作为独立实验结果。待 30 张盲测通过后，再决定是否组合“binary 存在性 + ordinal 严重度”。

## 7. 策略变化

- binary injury 作为独立校准证据，不覆盖四类 severity。
- binary 阴性时否决四分类高血腥候选。
- binary 与风险头或“后果展示”高分冲突时，动作改为 `人工复核`。
- LLM 外部识图默认关闭，仅保留显式审计开关。

## 8. 复现

```bash
python -B -m unittest discover -s tests

python -B -m src.evaluate_hard_probe \
  --manifest data/hard_probe/manifest.csv \
  --image-root pic/pic \
  --config config/inference.yaml \
  --out eval_results/hard_probe_baseline.json

python -B -m src.train_binary_injury \
  --config config/train.yaml \
  --checkpoint checkpoints/best \
  --hard-negatives data/train_hard_negatives.csv \
  --output-dir checkpoints/best/injury_binary \
  --eval-out eval_results/binary_injury_metrics.json

python -B -m src.evaluate_hard_probe \
  --manifest data/hard_probe/manifest.csv \
  --image-root pic/pic \
  --config config/inference.yaml \
  --binary-head checkpoints/best/injury_binary \
  --out eval_results/hard_probe_binary.json

# CORAL ordinal head
python -B -m src.train_ordinal_injury \
  --config config/train.yaml \
  --checkpoint checkpoints/best \
  --hard-negatives data/train_hard_negatives.csv \
  --output-dir checkpoints/injury_ordinal \
  --eval-out eval_results/ordinal_injury_metrics.json

python -B -m src.evaluate_hard_probe \
  --manifest data/hard_probe/manifest.csv \
  --image-root pic/pic \
  --config config/inference.yaml \
  --ordinal-head checkpoints/injury_ordinal \
  --out eval_results/hard_probe_ordinal.json
```

## 9. 后续

1. 完成 45 张有血图的轻/中/重标注。
2. 使用 30 张独立盲测图验证当前 binary 结果。
3. 对文字扭曲、严重模糊和遮挡建立独立质量探针。
4. 评估序数损失和局部 token 注意力，而不是优先扩大 LoRA。
5. 将 RAG 限定为法规依据与语境例外，不用于视觉血腥判断。

## 10. Label Studio severity 回填

Label Studio 配置：`config/label_studio_hard_probe_injury.xml`。

只需导入 45 张有血图，标注三种 `injury_detail`：

- `轻微（微量血迹/淤青）`
- `中度（明显开放性伤口/大量血泊）`
- `严重（器官/骨骼暴露/分尸）`

导出后回填 manifest：

```bash
python -m src.merge_hard_probe_severity \
  --manifest data/hard_probe/manifest.csv \
  --export path/to/label-studio-export.json \
  --out data/hard_probe/manifest.csv
```

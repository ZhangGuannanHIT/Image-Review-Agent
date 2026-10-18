# 训练报告（2026-10-18）

本次使用 **FG-CLIP2 视觉塔 + LoRA + 四任务分类头**，在清洗后的 2922 条数据上完成第一轮正式训练，并产出可加载的 checkpoint。结论：训练链路和权重可用；风险等级与暴力类型已有明显区分能力；`quality`（尤其"文字扭曲形变"）仍是短板，需要更多标注和后续调优。

## 1. 训练配置

| 项目 | 值 |
|---|---|
| 基础模型 | `fg_clip`（`/home/ygx/fg_clip`，1.5GB） |
| 微调方式 | 视觉塔 LoRA：r=16, alpha=32, dropout=0.05, target=q/k/v/out_proj；文本塔冻结 |
| 分类头 | 共享层 Linear(768,512)+GELU+Dropout，4 个头（5/4/4/5 维） |
| 损失 | risk 1.0 + type 0.5 + injury 0.5 + quality 0.5；label_smoothing=0.05 |
| 优化器 | AdamW；LoRA lr=1e-4，head lr=1e-3，weight_decay=0.01 |
| batch | **4 × grad_accum 4（有效 batch 16）** |
| epoch / 早停 | 最多 20 / patience 5，指标为验证集 risk macro-F1 |
| AMP | fp16 |
| 环境 | WSL Ubuntu / conda `fgclip`，Python 3.10.21，torch 2.5.1+cu124，peft 0.21.2 |
| GPU | RTX 4060 Laptop 8GB，训练时占用约 3.6–3.8GB，利用率 65–95% |
| 耗时 | 约 14 分钟（9 个 epoch） |

> batch 从 16×1 改成 4×4：8GB 显存下 16 张图（多数 784 patch）同时前向/反向风险过高；4×4 保持有效 batch 不变。

## 2. 数据划分

按 risk 分层划分，固定写入 `data/split.json`：

| split | 样本 | L0 | L1 | L2 | L3 | X | risk 缺失 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2483 | 1252 | 470 | 508 | 110 | 133 | 10 |
| val | 292 | 147 | 55 | 60 | 13 | 16 | 1 |
| test | 147 | 74 | 28 | 30 | 7 | 7 | 1 |

多标签正样本数（train / val / test）：

- type：无暴力 1344/163/80，直接暴力 676/78/39，后果展示 242/25/15，暗示威胁 237/22/12。
- quality：清晰 613/79/47，轻微模糊 970/111/52，严重模糊 899/100/48，文字扭曲 170/15/14，故意遮挡 **17/3/3**。

注意：`故意遮挡` 和 `文字扭曲` 的测试正样本只有 3 条和 14 条，相关指标波动很大。

## 3. 训练过程

验证集 risk macro-F1：

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro-F1 | 0.4459 | 0.4846 | 0.4942 | **0.5289** | 0.5108 | 0.4831 | 0.5169 | 0.4955 | 0.5145 |

第 4 轮最好，之后连续 5 轮未提升，第 9 轮触发早停。`checkpoints/best` 保存的是第 4 轮权重。

## 4. 验证集指标（`checkpoints/best/metrics.json`）

| 任务 | 指标 | 值 |
|---|---|---:|
| risk | accuracy / macro-F1 / L3 recall | 0.6254 / **0.5289** / 0.7692 |
| type | micro-F1 / macro-F1 | 0.7452 / 0.6731 |
| injury | accuracy / macro-F1 | 0.8201 / 0.5193 |
| quality | micro-F1 / macro-F1 | 0.5935 / 0.4206 |

type 逐类 F1：无暴力 0.833、直接暴力 0.750、后果展示 0.595、暗示威胁 0.514。
quality 逐类 F1：清晰 0.667、轻微模糊 0.599、严重模糊 0.690、文字扭曲 0.148、故意遮挡 0。

## 5. 测试集指标（`eval_results/`）

单标签头（risk/injury）不受阈值影响：

| 任务 | 指标 | 值 |
|---|---|---:|
| risk | accuracy / macro-F1 / L3 recall | 0.7123 / **0.6215** / **0.8571** |
| injury | accuracy / macro-F1 | 0.8116 / 0.4614 |

多标签头在默认 0.5 与标定阈值下的对比：

| 任务 | 指标 | 默认 0.5 | 标定后 |
|---|---|---:|---:|
| type | micro-F1 / macro-F1 | 0.8051 / 0.7287 | **0.8224 / 0.7757** |
| quality | micro-F1 / macro-F1 | 0.6020 / 0.4500 | 0.5796 / **0.4556** |

标定后 type 逐类：无暴力 P0.886/R0.875/F0.881；直接暴力 P0.750/R0.846/F0.795；后果展示 P0.560/R0.933/F0.700；暗示威胁 P0.800/R0.667/F0.727。

标定后 quality 逐类：清晰 P0.698/R0.787/F0.740；轻微模糊 P0.411/R0.981/F0.580；严重模糊 P0.532/R0.854/F0.656；文字扭曲 P0.181/R0.929/F0.302；故意遮挡 0（测试仅 3 个正样本，模型完全没学会）。

quality 的 micro-F1 略降、macro-F1 略升，是召回优先阈值的正常结果：多找回了一些"轻微模糊/文字扭曲"，代价是误报增加。

## 6. 阈值标定

在验证集（292 条）上逐类扫描 0.05–0.95，按 F1 选阈值，写入 `config/inference.yaml`：

| 类别 | 阈值 | 依据 |
|---|---:|---|
| 无暴力 | 0.45 | F1 最优 |
| 直接暴力 | 0.60 | F1 最优 |
| 后果展示 | 0.60 | F1 最优 |
| 暗示/威胁 | 0.90 | F1 最优（测试集 F1 从 0.50 提升到 0.727） |
| 清晰 | 0.35 | F1 最优 |
| 轻微模糊 | 0.30 | F1 最优 |
| 严重模糊 | 0.55 | F1 最优 |
| 文字扭曲形变 | **0.25（人工覆盖）** | F1 最优是 0.7，但召回只有 40%；项目重点是隐性违规，改为召回优先，验证集召回 87%、测试集召回 92.9% |
| 故意遮挡 | 0.50 | 验证集仅 3 个正样本，未标定 |

完整曲线与复算指标见 `eval_results/threshold_calibration.json`，可重跑：

```bash
python -m src.calibrate_thresholds --config config/train.yaml \
  --checkpoint checkpoints/best --split val --test-split test \
  --override "文字扭曲形变=0.25" --out eval_results/threshold_calibration.json
```

## 7. 指标怎么读

- **accuracy**：所有样本里猜对的比例。类别不均衡时会虚高，不能单独看。
- **macro-F1**：每个类各算一个 F1 再平均，小类（L3、X）权重和大类一样，是风险等级最主要的指标。
- **micro-F1**（多标签）：把所有类别的是/否判断合在一起算，受大类影响更大。
- **precision / recall**：precision 是"报出来的有多少是真的"，recall 是"真的有多少被报出来"。内容审核通常更怕漏报，所以会牺牲 precision 换 recall。
- **混淆矩阵**：`行=真实等级，列=预测等级`。例如 L3 行只有 7 条，其中 6 条预测正确，就是 `L3 recall=0.857`。
- **L3 recall**：严重违规的召回率，是业务上最敏感的单指标。
- **trained=true**：表示 LoRA adapter + head 都已成功加载；随机初始化头时为 false，结果不能当效果。

## 8. 本次训练的限制

1. 数据量小：2922 条，L3 只有 130 条，X 只有 156 条，测试集 L3 只有 7 条；指标置信区间很宽。
2. 标注噪声：106 条样本 `violence_type=无暴力` 但 `injury_detail` 有血腥（其中 30 条 L3），这类矛盾会拖累 type/injury 两个头。
3. `故意遮挡` 全量只有 23 条（验证/测试各 3 条），当前模型基本没有该能力。
4. `文字扭曲` 只有 199 条，且视觉上常与"严重模糊"共存，单标签区分度还不够。
5. 只训练了 9 个 epoch 就早停，继续加数据比继续调超参更有效。

## 9. 产物

```text
checkpoints/best/
├── adapter_config.json
├── adapter_model.safetensors    # LoRA 权重（约 4.8MB）
├── head.pt                      # 四任务分类头（约 1.6MB）
├── labels.json                  # schema_version 2 + task_types
├── metrics.json                 # 验证集指标
└── README.md                    # 自动生成的说明
eval_results/
├── test_metrics.json            # 默认 0.5 阈值下的测试集指标
└── threshold_calibration.json   # 阈值曲线与标定后指标
logs/
├── train_20261018.log
└── test_eval_20261018.log
```

## 10. 下一步建议

1. 优先补 L3、X、故意遮挡、文字扭曲的标注，这四类的提升空间最大。
2. 修掉"无暴力 + 有血腥"的矛盾标注后重训一轮，type/injury 应能明显改善。
3. 数据翻倍后重新标定阈值，并把 `config/inference.yaml` 的阈值视为"当前版本"而不是永久值。
4. 接后端时把 `backend/config.yaml` 改成 `lora: true`、`require_checkpoint: true`、`ckpt_dir: checkpoints/best`。
5. 用 Agent 端到端评测（人工抽检 100–200 张）验证"CLIP 证据 + LLM 裁决"的最终准确率，而不只看分类头指标。

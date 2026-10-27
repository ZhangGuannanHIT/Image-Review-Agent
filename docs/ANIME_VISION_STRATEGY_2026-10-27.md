# 动漫专用策略与 LLM 视觉复核（2026-10-27）

## 1. 分流结构

```text
FG-CLIP2 + binary + quality
  → anime style router
      ├─ 真实/低置信：现有真实策略，不调用动漫视觉
      └─ 高置信动漫：effective risk
           ├─ allow：直接放行
           └─ limit/review/block：条件触发 LLM 视觉复核
```

风格路由 checkpoint：

```text
checkpoints/anime_style_router/
  router.pt
  metadata.json
```

## 2. 风格路由

训练数据：

- 动漫/游戏正例：120 张 Label Studio 已标注图片；
- 真实负例：solid real 100 张 + quality real 60 张；
- 冻结 FG-CLIP2 768 维 embedding；
- 轻量 768→128→1 网络。

确定性划分：

- train 181
- val 63
- test 36

生产阈值 `0.90`：

| split | 动漫召回 | 真实误路由 |
|---|---:|---:|
| train | 100% | 0% |
| validation | 100% | 0% |
| test | 100% | 0% |

该结果非常理想，主要因为当前真实负例比较典型，不能当作开放互联网泛化证明。
低置信图片默认走真实策略，避免动漫策略污染真实域。

## 3. LLM 视觉触发

同时满足才触发：

1. 风格路由高置信判定为动漫；
2. 初步动作是 `limit/review/block`；
3. 没有 grounded 文字违规；
4. 高风险 RAG 命中没有被文字语义复核判为 safe。

动作是 `allow` 或真实域时不触发。

360 张当前动漫候选中：

- 初步动作：187 allow、110 limit、63 review；
- 预计视觉触发：171 张；
- 未触发：189 张。

触发数量仍然较高，线上需要关注 API 成本和延迟。

## 4. 提示词

提示词沿用远端结构化思路，但针对动漫域增加：

- `weapon_visible`；
- `context`：game_ui、game_gameplay、animation、illustration、real_photo；
- 战斗 pose、技能特效、握拳冲刺但无接触/无目标时不判 attack；
- 红色染料、红色背景、特效、受伤妆和卡通红痕不直接判 injury；
- 皮肤、衣服、地面上的明显红色喷溅/流淌，在无明确颜料说明时可判 mild；
- 持刀/持枪站立、产品展示不等于 threat；
- threat=present 必须有对象和明确武器关系；
- FPS 游戏界面、武器皮肤展示、游戏内攻击动作属于游戏语境。

## 5. 程序复核

LLM 不能返回最终动作。

| 视觉证据 | 程序结果 |
|---|---|
| 明确游戏语境，无 injury，无 threat | allow |
| 动画虚构语境，仅 attack，无 injury/threat | allow |
| 只有 weapon_visible，非游戏语境 | L1 limit |
| injury mild / uncertain、attack/threat/weapon uncertain | L1 limit，不转人工 |
| 暗黑/horror 风格且无明确伤害关系 | L1 limit |
| injury moderate/severe、attack present、threat present | 保留原动作 |
| grounded 文字违规 | 不撤回原动作 |
| 视觉调用失败或 JSON 无效 | 保留原动作 |

## 6. 在线小样本实测

15 张分层真实 API 样本：

- 10 张已确认安全动漫：10 放行；
- 2 张已确认 L1：2 限流；
- 3 张已知 AI 出血正样本：3 限流；
- 视觉调用失败：0。

该样本用于调试边界，不是独立总体指标。

真实域回归：

- solid 180：动作分布与接入前完全一致；
- real 100：0 违规自动放行；
- quality real：违规自动放行 3 张，与 v4 基线相同，没有新增。

## 7. 已确认标注

Label Studio 导入 120/120：

- 初版 risk：115 L0、2 L1、2 L2、1 L3；
- 人工确认 3 张 L2/L3 为误选，修正为 L0；
- FPS 游戏武器展示 2 张也确认允许，修正为 L0；
- 最终 120 张全部 L0；
- 加上之前 6 张，当前共确认 126 张：124 L0、2 L1。

## 8. 局限

- 风格路由负例分布仍偏典型；
- 仅有 3 张出血图做了在线视觉实测；
- 未对 360 张全量执行外部视觉模型，171 次调用的费用/延迟未测；
- 提示词对罕见武器关系、极端遮挡和真实 AI 出血仍需扩展测试；
- 视觉模型失败时不会撤回候选，因此不会降低安全底线，但可能继续偏保守。

## 9. 可移植性

- 视觉模型必须支持 OpenAI 兼容的 `image_url` 消息；当前 DeepSeek 配置已实测可调用。
- 如果 Provider 不支持图片，调用返回 `failed`，程序保留原动作，不会误放行。
- 路由 checkpoint 由仓库内 `checkpoints/anime_style_router/` 提供；缺失时回退到
  场景 Prompt 域证据。
- 训练脚本的图片根目录、标签清单和输出目录均由 CLI 参数配置，不依赖固定 Windows 盘符。
- 视觉触发、域阈值和最大图片边长均可在 `backend/config.yaml` 调整。

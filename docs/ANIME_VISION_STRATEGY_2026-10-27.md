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

满足以下任一视觉路由，再通过文本安全门控才触发：

1. 风格路由高置信判定为动漫；或
2. `game_ui_weapon_display` 冻结 CLIP Prompt 分数 `>= 0.039`，且类型头命中武器。

路由优先级：动漫风格路由先判定；只有动漫路由未命中时，游戏 UI 候选才作为兜底。
因此真实照片或写实游戏画面不会仅因“没通过动漫头”而被当成动漫；但武器类游戏 UI
仍可通过窄候选进入视觉复核。

共同条件：

- 初步动作是 `limit/review/block`；
- 没有 grounded 文字违规；
- 高风险 RAG 命中没有被文字语义复核判为 safe。

动作是 `allow` 时只在既有安全信号下触发。普通真实图不会仅因风格为 real 触发；
但 3D 游戏界面可能在 style router 中被判为 real，因此允许游戏 UI 候选旁路该判断。
该旁路只增加视觉复核，不直接改变动作；视觉返回非游戏语境时保留原动作。

游戏 UI Prompt 使用冻结 FG-CLIP2 图文相似度，不需要新增图片或训练新模型：

- Prompt：4 条，覆盖武器皮肤预览、游戏菜单/物品栏、FPS HUD/准星/弹药/小地图；
- 阈值：`0.039`；
- 输出：`routing_signal`、`game_ui_score`、`game_ui_matched`；
- 独立于原有 `risk/safe_context/confounder/domain` 分数，不改变原策略动作证据。

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

2026-10-29 游戏 UI 路由回归：

- 两张游戏武器皮肤页：均从 L2 撤回 L0；
- `ANIME_SAFE_1067` 真实持刀图：保持 L1 限流；
- `OIP1 (17)` 暗黑插画：保持 L1 限流；
- Solid/Quality 共 300 张扫描新增 3 个候选：1 张 FPS 血迹保持 L3 limit；
  1 张真实射击照片返回 real_photo，保持 L2 limit；1 张持刀威胁插画返回 animation，
  保持 L2 limit；
- 新增真实域违规自动放行：0。

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

## 10. 参数优化对照（2026-10-28）

虚拟视觉 Prompt 和输出 schema 保持不变。独立视觉模型使用：

- `max_tokens=512`
- `max_retries=1`
- OpenAI-compatible JSON response mode
- `thinking=disabled`
- 显式检查 `finish_reason=length`

DeepSeek JSON mode 会额外返回 `"type":"json_object"` 控制字段。解析器只精确移除这个
字段，其他未知字段仍由 Pydantic 拒绝。

8 张分层样本、1280/960 各一次，共 16 次调用全部成功：

| 指标 | 结果 |
|---|---:|
| 1280 平均耗时 | 1371.9 ms |
| 960 平均耗时 | 1340.1 ms |
| `injury` 一致 | 8/8 |
| `attack` 一致 | 8/8 |
| `threat` 一致 | 8/8 |
| `weapon_visible` 一致 | 7/8 |
| `context` 一致 | 7/8 |
| `appearance` 一致 | 8/8 |
| `threat_basis` 一致 | 8/8 |

差异只出现在武器可见性和 `game_ui/game_gameplay` 子类型。两张样本最终均为放行，
但字段并非完全一致，因此默认分辨率继续使用 1280，960 只保留为实验参数。

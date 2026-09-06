# -*- coding: utf-8 -*-
"""标签常量与字段定义——数据集/训练/评估/推理/后端共用的唯一口径。

依据：标注 JSON（Label Studio 导出）的 from_name 控件枚举值。
新增类别或调整枚举时只改本文件，勿在别处硬编码。
"""

# ==== risk_level（主标签，choices 单选，含独立"模糊"类 X）====
# 类名 = Label Studio 枚举值前缀（值常带括号注释，匹配时取前缀）。
# X_模糊 对应 JSON 值 "X_无法判定（严重模糊/扭曲导致内容不可辨）"。
RISK_CLASSES = [
    "L0_安全", "L1_轻微违规", "L2_中度违规", "L3_严重违规", "X_无法判定",
]  # 索引即 label id；X 是独立第 5 类

# 标签含义（用于生成可读文本 / LLM 提示词）
RISK_LABEL_TEXT = {
    "L0_安全": "正常内容，无违规",
    "L1_轻微违规": "轻微违规（轻度暴力/少量血迹）",
    "L2_中度违规": "中度违规（明显暴力/开放性伤口）",
    "L3_严重违规": "严重违规（极端血腥/虐杀/儿童受害）",
    "X_无法判定": "无法判定（图中内容难以辨认）",
}

# ==== violence_type（taxonomy 多选，按集合去序，4 路独立二分类）====
# 类名 = JSON 枚举中的" / "符号保留，确保与 Label Studio 值对齐。
TYPE_CLASSES = [
    "无暴力", "直接暴力（攻击进行中）", "后果展示（尸体/伤口）", "暗示/威胁（武器指向/捆绑）",
]  # 可多选

# ==== injury_detail（choices 单选，4 档）====
INJURY_CLASSES = [
    "无血腥", "轻微（微量血迹/淤青）", "中度（明显开放性伤口/大量血泊）", "严重（器官/骨骼暴露/分尸）",
]

# ==== quality（taxonomy 单选，5 类）====
QUALITY_CLASSES = [
    "清晰", "轻微模糊", "严重模糊", "文字扭曲形变", "故意遮挡（马赛克/黑条/贴纸）",
]

# 缺失标记（用于 ignore_index / mask）
IGNORE_INDEX = -1

# 训练总损失权重
LOSS_WEIGHTS = {"risk": 1.0, "type": 0.5, "injury": 0.5, "quality": 0.5}


def validate_label_schema() -> None:
    """校验训练、推理和 Agent 共用的标签契约。"""
    groups = {
        "risk": RISK_CLASSES,
        "type": TYPE_CLASSES,
        "injury": INJURY_CLASSES,
        "quality": QUALITY_CLASSES,
    }
    for name, values in groups.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} 标签为空或存在重复值: {values}")


validate_label_schema()

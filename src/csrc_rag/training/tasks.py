"""训练任务定义：声明式描述「处罚类型多标签预测」任务的输入输出契约。

我们把任务定义抽象成一个不可变的 ``TrainingTask`` 元数据对象，集中声明：
预测目标字段、允许使用的输入字段、**必须屏蔽**的字段、以及评估指标集。
这样做的关键意义在于「防标签泄漏」——处罚结果相关字段（如处罚类型本身、
处罚措施、罚没金额）绝不能进入特征，否则模型会「看着答案预测答案」，
评测虚高却毫无实际价值。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingTask:
    # 任务契约（不可变）：把「预测什么、能看什么、不能看什么、怎么评」固化为配置，
    # 让数据构造和训练流程都以此为唯一事实来源。
    name: str
    target_field: str
    allowed_input_fields: list[str]   # 允许作为特征的字段（违规行为、法条、当事人画像等）
    blocked_input_fields: list[str]   # 强制屏蔽字段，防止处罚结果信息泄漏进输入
    metrics: list[str]


# 处罚类型多标签预测任务实例：输入仅含违规事实与主体信息，
# 屏蔽一切处罚结果字段，评估覆盖 Micro/Macro-F1、汉明损失与子集精确率。
PUNISHMENT_TYPE_TASK = TrainingTask(
    name="punishment_type_multilabel",
    target_field="PunishmentType",
    allowed_input_fields=[
        "Activity",
        "Law",
        "Party",
        "Position",
        "Relationship",
        "Promulgator",
        "DeclareDate",
        "IsListedCom",
    ],
    blocked_input_fields=[
        "PunishmentType",
        "PunishmentMeasure",
        "SumPenalty",
    ],
    metrics=["Micro-F1", "Macro-F1", "HammingLoss", "SubsetAccuracy"],
)


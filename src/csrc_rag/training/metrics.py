"""多标签评估指标：统一计算 Micro-F1 / Macro-F1 / 汉明损失 / 子集精确率。

这是 sklearn 基线和 HF 微调共用的指标实现，保证两条训练路线用完全一致的口径对比效果：
* Micro-F1：把所有标签的 TP/FP/FN 汇总后算 F1，侧重整体（受高频标签影响大）。
* Macro-F1：逐标签算 F1 再平均，对低频标签更公平。
* HammingLoss：逐标签预测错误的平均比例，越低越好。
* SubsetAccuracy：整行标签**完全匹配**才算对，是最严格的指标。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # 子集精确率：要求一条样本的整个标签向量与真值逐位相等才计为正确。
    subset_accuracy = float(np.mean(np.all(y_true == y_pred, axis=1)))
    # zero_division=0：某标签在该批次完全未出现时，F1 记 0 而非抛警告，保证数值稳定。
    return {
        "Micro-F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "HammingLoss": float(hamming_loss(y_true, y_pred)),
        "SubsetAccuracy": subset_accuracy,
    }


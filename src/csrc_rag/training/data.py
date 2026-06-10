"""训练数据加载与切分：JSONL 读取、按时间切分、标签词表构建与多标签编码。

本模块是训练库的数据底座，所有下游训练器（sklearn 基线、HF 微调）共用这里的工具：
* 数据按「公告时间」升序排列，再做**时间切分**而非随机切分——这样测试集永远是
  训练集「之后」的案例，避免用未来信息预测过去造成的数据泄漏，更贴近真实上线场景。
* 标签词表统一在三套数据上构建，保证多标签向量的维度与索引在各集合间一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from csrc_rag.utils import read_jsonl


@dataclass(frozen=True)
class SplitDataset:
    # 训练/验证/测试三分数据集的不可变容器。
    train: list[dict[str, Any]]
    valid: list[dict[str, Any]]
    test: list[dict[str, Any]]


def _parse_date(value: str | None) -> datetime:
    # 解析公告日期用于排序；缺失或格式异常时统一回退到 1900-01-01，
    # 让这类样本稳定排到最前，避免排序键报错或顺序抖动。
    if not value:
        return datetime(1900, 1, 1)
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d")
    except ValueError:
        return datetime(1900, 1, 1)


def load_party_samples(path: str) -> list[dict[str, Any]]:
    # 读取 JSONL 样本并按（公告日期, sample_id）升序稳定排序，
    # sample_id 作为同日样本的二级键，保证排序结果可复现。
    rows = read_jsonl(path)
    rows.sort(key=lambda row: (_parse_date(row.get("declare_date")), row.get("sample_id")))
    return rows


def time_split(rows: list[dict[str, Any]], train_ratio: float = 0.7, valid_ratio: float = 0.15) -> SplitDataset:
    # 时间切分：在已按时间排好序的样本上按比例切前 70% 训练、中 15% 验证、尾 15% 测试。
    # 因为输入已是时间有序的，这里的顺序切分天然实现了「用过去预测未来」的评估设定。
    n = len(rows)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return SplitDataset(train=rows[:train_end], valid=rows[train_end:valid_end], test=rows[valid_end:])


def build_label_vocab(rows: list[dict[str, Any]]) -> list[str]:
    # 汇总所有样本出现过的标签并去重排序，得到稳定有序的标签词表（决定后续向量列序）。
    labels = sorted({label for row in rows for label in row.get("labels", [])})
    return labels


def encode_multilabel(rows: list[dict[str, Any]], vocab: list[str]) -> np.ndarray:
    # 把每条样本的标签列表编码成 multi-hot 0/1 向量：命中的标签对应列置 1，其余为 0。
    label_to_idx = {label: idx for idx, label in enumerate(vocab)}
    target = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for label in row.get("labels", []):
            target[row_idx, label_to_idx[label]] = 1.0
    return target


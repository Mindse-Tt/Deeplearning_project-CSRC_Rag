"""sklearn 传统机器学习基线：TF-IDF + 一对多逻辑回归的多标签分类。

本模块提供处罚类型预测任务的**对照基线**。我们刻意用经典、轻量、可解释的
TF-IDF 特征叠加 One-vs-Rest 逻辑回归，作为衡量 HF 微调模型「是否真的值得上深度学习」
的参照系——若深度模型相对这条基线没有明显增益，那就说明任务本身或数据更值得先打磨。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from csrc_rag.training.data import build_label_vocab, encode_multilabel
from csrc_rag.training.metrics import multilabel_metrics


@dataclass(frozen=True)
class SklearnTrainingOutput:
    # 基线训练产物：测试集指标 + 标签词表（便于与 HF 路线对齐标签口径）。
    metrics: dict[str, float]
    label_vocab: list[str]


def train_tfidf_logreg(
    train_rows: list[dict],
    valid_rows: list[dict],
    test_rows: list[dict],
) -> SklearnTrainingOutput:
    """训练 TF-IDF + OvR-LogReg 基线并在测试集上评估。"""
    # 标签词表跨三套数据统一构建，确保与 HF 路线的标签维度/索引一致、指标可直接对比。
    vocab = build_label_vocab(train_rows + valid_rows + test_rows)
    # TF-IDF 取 1~2 元语法、上限 5 万维特征，兼顾词与短语信息又控制特征膨胀。
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

    # 关键纪律：向量器只在训练集上 fit，测试集仅 transform，杜绝测试信息泄漏。
    x_train = vectorizer.fit_transform([row["input_text"] for row in train_rows])
    x_test = vectorizer.transform([row["input_text"] for row in test_rows])

    y_train = encode_multilabel(train_rows, vocab)
    y_test = encode_multilabel(test_rows, vocab)

    # One-vs-Rest：为每个标签独立训练一个二分类器，天然适配多标签场景；
    # liblinear 求解器在高维稀疏 TF-IDF 特征上收敛快且稳定。
    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="liblinear")
    )
    classifier.fit(x_train, y_train)
    # 输出每个标签的概率，再以 0.5 阈值二值化得到多标签预测，最后用统一指标评估。
    y_prob = classifier.predict_proba(x_test)
    y_pred = (y_prob >= 0.5).astype(np.float32)
    metrics = multilabel_metrics(y_test, y_pred)
    return SklearnTrainingOutput(metrics=metrics, label_vocab=vocab)


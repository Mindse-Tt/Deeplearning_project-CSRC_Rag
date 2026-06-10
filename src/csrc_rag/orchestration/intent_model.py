"""L1 意图分类层 —— TF-IDF + 逻辑回归（Logistic Regression）轻量路由器。

本模块是七层 RAG 流水线最前端的意图识别组件，负责把用户原始 query 映射到
7 个意图标签之一：``greeting / chitchat / out_of_scope / case_retrieval /
law_grounding / sanction_recommendation / trend_analysis``。

设计思路（我们刻意选用经典 ML 而非大模型分类）：
  * 字符级 TF-IDF（``char_wb`` + 2-4 gram）对中文短文本鲁棒，无需分词依赖；
  * 逻辑回归推理在毫秒级，适合放在流水线最前端做高频路由；
  * 训练侧用前后缀模板做数据增强，弥补人工标注样本稀缺；
  * 产物以 pickle 持久化到 ``artifacts/``，服务侧通过自定义 Unpickler 加载，
    使训练脚本与在线服务彻底解耦（服务侧不引入任何训练期依赖）。

本文件同时承载 v1 与 v2 两套分类器：v2 是我们后续扩展的 7 分类 Planner 版本，
两者接口（``predict``）保持一致，下游调用方可统一对待。
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from csrc_rag.settings import ARTIFACTS_DIR, CONFIG_DIR


DEFAULT_INTENT_ARTIFACT = ARTIFACTS_DIR / "intent_classifier" / "intent_model.pkl"
DEFAULT_INTENT_REPORT = ARTIFACTS_DIR / "intent_classifier" / "intent_report.json"


@dataclass(frozen=True)
class IntentPrediction:
    # 单次预测结果的不可变载体：name 为最高分意图，scores 保留全类别概率分布，
    # confidence 为该意图概率，method 记录预测来源（便于 trace 与离线评估）。
    name: str
    confidence: float
    scores: dict[str, float]
    method: str


@dataclass(frozen=True)
class IntentPredictionV2:
    """Prediction record emitted by the v2 (7-class) Planner classifier.

    Kept as a distinct dataclass so the pickle schema stored under
    ``artifacts/intent_classifier_v2/`` can be rehydrated without importing
    the training script. Shares the same duck-typed shape as
    ``IntentPrediction`` to remain compatible with downstream consumers.
    """

    name: str
    confidence: float
    scores: dict[str, float]
    method: str


class TfidfIntentClassifier:
    def __init__(self, vectorizer: TfidfVectorizer, classifier: LogisticRegression, labels: list[str]) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.labels = labels

    def predict(self, query: str) -> IntentPrediction:
        # 推理流程：query → TF-IDF 向量 → LogReg 概率分布 → 取 argmax 作为意图标签。
        features = self.vectorizer.transform([query])
        probabilities = self.classifier.predict_proba(features)[0]
        # 按概率降序整理为 {label: prob}，下游可据此做置信度兜底（见 reject_policy）。
        scores = {
            label: float(probability)
            for label, probability in sorted(
                zip(self.classifier.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        label = max(scores.items(), key=lambda item: item[1])[0]
        return IntentPrediction(
            name=label,
            confidence=round(scores[label], 4),
            scores={key: round(value, 4) for key, value in scores.items()},
            method="tfidf_logistic_regression",
        )


class TfidfIntentClassifierV2:
    """V2 Planner classifier (7 classes): ``greeting / chitchat / out_of_scope
    / case_retrieval / law_grounding / sanction_recommendation / trend_analysis``.

    Mirrors :class:`TfidfIntentClassifier` so callers can treat both uniformly.
    The class is re-declared here (not imported from the training script) so
    the v2 pickle can be loaded by the serving layer without pulling in any
    training-time dependencies.
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        classifier: LogisticRegression,
        labels: list[str],
    ) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.labels = labels

    def predict(self, query: str) -> IntentPrediction:
        features = self.vectorizer.transform([query])
        probabilities = self.classifier.predict_proba(features)[0]
        scores = {
            label: float(probability)
            for label, probability in sorted(
                zip(self.classifier.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )
        }
        label = max(scores.items(), key=lambda item: item[1])[0]
        return IntentPrediction(
            name=label,
            confidence=round(scores[label], 4),
            scores={key: round(value, 4) for key, value in scores.items()},
            method="tfidf_logistic_regression_v2",
        )


class _IntentV2Unpickler(pickle.Unpickler):
    """Unpickler that rehydrates the v2 artifact regardless of origin module.

    The v2 pickle was produced by ``scripts/train_intent_classifier_v2.py`` and
    references ``TfidfIntentClassifierV2`` / ``IntentPredictionV2`` under the
    ``__main__`` (or training script) module. At serving time neither module is
    importable, so we map those names onto the local re-declarations above.
    """

    _SHIMS = {
        "TfidfIntentClassifierV2": TfidfIntentClassifierV2,
        "IntentPredictionV2": IntentPredictionV2,
        "TfidfIntentClassifier": TfidfIntentClassifier,
        "IntentPrediction": IntentPrediction,
    }

    def find_class(self, module: str, name: str) -> Any:  # type: ignore[override]
        if name in self._SHIMS:
            return self._SHIMS[name]
        return super().find_class(module, name)


def _load_model_config() -> dict[str, Any]:
    config_path = CONFIG_DIR / "models.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _resolve_artifact_path(explicit: str | Path | None) -> Path:
    """Resolve the intent artifact path, honouring ``configs/models.json``.

    Priority: explicit caller argument > ``intent_router.artifact_path`` in
    models.json > legacy v1 default. This keeps the serving side configurable
    without editing Python sources.
    """
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = (CONFIG_DIR.parent / candidate).resolve()
        return candidate

    cfg = _load_model_config().get("intent_router", {}) or {}
    raw = cfg.get("artifact_path")
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (CONFIG_DIR.parent / candidate).resolve()
        return candidate
    return DEFAULT_INTENT_ARTIFACT


def load_examples(path: str | Path | None = None) -> tuple[list[str], list[str]]:
    config_path = Path(path) if path else CONFIG_DIR / "intent_examples.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    texts: list[str] = []
    labels: list[str] = []
    for label, examples in payload.items():
        for text in examples:
            texts.append(text)
            labels.append(label)
    return texts, labels


def _augment_texts(texts: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    # 数据增强：用常见口语前缀/后缀对每条样本做笛卡尔积扩展，
    # 让分类器对真实用户的多样化措辞更鲁棒（人工标注样本有限的弥补手段）。
    prefixes = ["", "请帮我", "帮我", "请问", "我想知道"]
    suffixes = ["", "。", "，请解释一下。", "，用于课程项目。"]
    augmented_texts: list[str] = []
    augmented_labels: list[str] = []
    for text, label in zip(texts, labels):
        for prefix in prefixes:
            for suffix in suffixes:
                candidate = f"{prefix}{text}{suffix}".strip()
                augmented_texts.append(candidate)
                augmented_labels.append(label)
    return augmented_texts, augmented_labels


def train_intent_classifier(
    examples_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    # 训练管线：读样本 → 增强 → 分层切分 → 拟合 → 评估 → 持久化产物 + 报告。
    texts, labels = load_examples(examples_path)
    texts, labels = _augment_texts(texts, labels)
    # stratify=labels 保证各意图在训练/测试集中的比例一致，避免小类被切空。
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    # 字符级 n-gram（char_wb）对中文无需分词即可建特征；sublinear_tf 抑制高频词。
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    # C=4.0 适度放宽正则以拟合增强后的稠密样本；max_iter 给足收敛余量。
    classifier = LogisticRegression(max_iter=2000, C=4.0)
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)
    classifier.fit(x_train_features, y_train)
    predictions = classifier.predict(x_test_features)
    report = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "num_train_examples": len(x_train),
        "num_test_examples": len(x_test),
        "labels": sorted(set(labels)),
    }

    model = TfidfIntentClassifier(vectorizer=vectorizer, classifier=classifier, labels=sorted(set(labels)))
    output_path = Path(artifact_path) if artifact_path else DEFAULT_INTENT_ARTIFACT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(model, handle)

    output_report = Path(report_path) if report_path else DEFAULT_INTENT_REPORT
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_intent_classifier(
    path: str | Path | None = None,
) -> TfidfIntentClassifier | TfidfIntentClassifierV2 | None:
    """Load either the v1 or v2 intent-classifier pickle.

    Behaviour:
        * When ``path`` is given, load directly from that file (explicit win).
        * Else read ``configs/models.json`` → ``intent_router.artifact_path``.
        * Else fall back to the legacy v1 default.

    V2 pickles were produced in the training script where the class lived in
    ``__main__`` (or the ``train_intent_classifier_v2`` module). Those names
    won't resolve at serving time, so :class:`_IntentV2Unpickler` shims them
    onto the locally re-declared :class:`TfidfIntentClassifierV2`.
    """
    model_path = _resolve_artifact_path(path)
    if not model_path.exists():
        return None
    with model_path.open("rb") as handle:
        return _IntentV2Unpickler(handle).load()


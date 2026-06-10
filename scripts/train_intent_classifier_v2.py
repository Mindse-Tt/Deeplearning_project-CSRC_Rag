"""Train the v2 Planner intent classifier (7 classes).

Classes (7)::

    greeting / chitchat / out_of_scope /
    case_retrieval / law_grounding / sanction_recommendation / trend_analysis

Backends:
    - sklearn  (default): TF-IDF char n-gram + LogisticRegression, CPU-friendly baseline.
    - fasttext (stub):    FastText zh subword model, requires the ``fasttext`` package.
    - bert     (stub):    DistilBERT/MiniLM-zh fine-tune, Colab/GPU required.

Typical usage::

    python scripts/train_intent_classifier_v2.py
    python scripts/train_intent_classifier_v2.py --backend sklearn
    python scripts/train_intent_classifier_v2.py \
        --examples configs/intent_examples_v2_augmented.json

Outputs::

    artifacts/intent_classifier_v2/intent_model_v2.pkl
    artifacts/intent_classifier_v2/intent_report_v2.json

The script does NOT touch the legacy v1 artifacts so both coexist during the
migration window. See ``docs/strategies/04-planner-training-strategy.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.settings import ARTIFACTS_DIR, CONFIG_DIR  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_EXAMPLES_V2 = CONFIG_DIR / "intent_examples_v2.json"
DEFAULT_ARTIFACT_V2 = ARTIFACTS_DIR / "intent_classifier_v2" / "intent_model.pkl"
DEFAULT_REPORT_V2 = ARTIFACTS_DIR / "intent_classifier_v2" / "intent_report.json"

EXPECTED_LABELS_V2: tuple[str, ...] = (
    "greeting",
    "chitchat",
    "out_of_scope",
    "case_retrieval",
    "law_grounding",
    "sanction_recommendation",
    "trend_analysis",
)

META_KEY = "_meta"


@dataclass(frozen=True)
class TrainingConfig:
    backend: str
    examples_path: Path
    artifact_path: Path
    report_path: Path
    test_size: float = 0.25
    random_state: int = 42
    min_df: int = 1
    ngram_range: tuple[int, int] = (2, 4)
    regularization_c: float = 4.0
    max_iter: int = 2000


@dataclass(frozen=True)
class IntentPredictionV2:
    name: str
    confidence: float
    scores: dict[str, float]
    method: str


class TfidfIntentClassifierV2:
    """TF-IDF + LogisticRegression classifier for the v2 (7-class) Planner."""

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        classifier: LogisticRegression,
        labels: list[str],
    ) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.labels = labels

    def predict(self, query: str) -> IntentPredictionV2:
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
        top_label = max(scores.items(), key=lambda item: item[1])[0]
        return IntentPredictionV2(
            name=top_label,
            confidence=round(scores[top_label], 4),
            scores={key: round(value, 4) for key, value in scores.items()},
            method="tfidf_logistic_regression_v2",
        )


def _load_jsonl_examples(path: Path) -> tuple[list[str], list[str]]:
    """Load ``{"text", "label"}`` per-line records (augmentation output)."""
    texts: list[str] = []
    labels: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text")
            label = record.get("label")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Record missing text: {record!r}")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Record missing label: {record!r}")
            texts.append(text.strip())
            labels.append(label.strip())
    return texts, labels


def _load_json_examples(path: Path) -> tuple[list[str], list[str]]:
    """Load the legacy JSON-dict format (``{label: [text, ...]}``)."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    texts: list[str] = []
    labels: list[str] = []
    for label, examples in payload.items():
        if label == META_KEY:
            continue
        if not isinstance(examples, list):
            raise ValueError(f"Intent '{label}' must map to a list of strings.")
        for text in examples:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Intent '{label}' contains an empty or non-string example.")
            texts.append(text.strip())
            labels.append(label)
    return texts, labels


def load_examples_v2(path: Path) -> tuple[list[str], list[str]]:
    """Load 7-class training data.

    Accepts both the legacy ``intent_examples_v2.json`` dict format and the
    augmented ``intent_train_v2.jsonl`` (one ``{"text","label"}`` per line).
    """
    if path.suffix.lower() == ".jsonl":
        texts, labels = _load_jsonl_examples(path)
    else:
        texts, labels = _load_json_examples(path)

    missing = set(EXPECTED_LABELS_V2) - set(labels)
    if missing:
        raise ValueError(f"Missing intents in examples file: {sorted(missing)}")
    return texts, labels


def augment_texts(texts: list[str], labels: list[str]) -> tuple[list[str], list[str]]:
    """Lightweight rule-based augmentation for seed-only training.

    When the input already contains ~500 LLM-augmented items per class,
    this still works but multiplies the set via prefix/suffix variants.
    """
    prefixes = ("", "请", "帮我", "请问", "我想知道", "麻烦")
    suffixes = ("", "。", "？", "，谢谢", "，用于课程项目")
    out_texts: list[str] = []
    out_labels: list[str] = []
    for text, label in zip(texts, labels):
        for prefix in prefixes:
            for suffix in suffixes:
                candidate = f"{prefix}{text}{suffix}".strip()
                out_texts.append(candidate)
                out_labels.append(label)
    return out_texts, out_labels


def train_sklearn_backend(config: TrainingConfig) -> dict[str, Any]:
    texts, labels = load_examples_v2(config.examples_path)
    # Seed-only files (~21 rows total) need rule-based expansion before training;
    # a fully augmented corpus (>=200 rows/class) already provides enough signal
    # and further multiplication hurts generalisation + slows training.
    per_class = min(Counter(labels).values())
    if per_class < 50:
        logger.info("Seed-only corpus detected (min=%d); applying augment_texts.", per_class)
        texts, labels = augment_texts(texts, labels)
    else:
        logger.info("Augmented corpus detected (min=%d); skipping augment_texts.", per_class)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.ngram_range,
        sublinear_tf=True,
        min_df=config.min_df,
    )
    classifier = LogisticRegression(
        max_iter=config.max_iter,
        C=config.regularization_c,
        class_weight="balanced",
    )

    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)
    classifier.fit(x_train_features, y_train)

    predictions = classifier.predict(x_test_features)
    class_labels = sorted(set(labels))
    cm = confusion_matrix(y_test, predictions, labels=class_labels).tolist()
    report = {
        "backend": "sklearn",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": class_labels,
            "matrix": cm,
        },
        "num_train_examples": len(x_train),
        "num_test_examples": len(x_test),
        "labels": class_labels,
        "examples_path": str(config.examples_path),
        "config": {
            "ngram_range": list(config.ngram_range),
            "regularization_c": config.regularization_c,
            "max_iter": config.max_iter,
            "test_size": config.test_size,
            "random_state": config.random_state,
        },
    }

    model = TfidfIntentClassifierV2(
        vectorizer=vectorizer,
        classifier=classifier,
        labels=class_labels,
    )
    config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with config.artifact_path.open("wb") as handle:
        pickle.dump(model, handle)

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def train_fasttext_backend(config: TrainingConfig) -> dict[str, Any]:
    """Stub for FastText backend (Phase 2).

    Planned steps:
        1. Dump examples to fastText format (``__label__<name> <text>``).
        2. ``fasttext.train_supervised(input=..., wordNgrams=2, minn=2,
           maxn=5, dim=100, epoch=25)``.
        3. Persist ``.bin`` artifact and dump macro-F1 via ``model.test``.

    See ``docs/strategies/04-planner-training-strategy.md`` section 6.
    """
    raise NotImplementedError(
        "fasttext backend is a Phase-2 stub; see strategy doc §6 for the plan."
    )


def train_bert_backend(config: TrainingConfig) -> dict[str, Any]:
    """Stub for DistilBERT / MiniLM-zh backend (Phase 3 ablation).

    Planned steps:
        1. Load a small zh encoder (e.g., ``uer/chinese-roberta-wwm-ext-tiny``).
        2. Fine-tune ``AutoModelForSequenceClassification`` with ``num_labels=7``.
        3. Run on Colab/Kaggle (本机无 GPU), export ONNX for CPU serving.

    See ``docs/strategies/04-planner-training-strategy.md`` section 6.
    """
    raise NotImplementedError(
        "bert backend is a Phase-3 stub; run on Colab per strategy doc §6."
    )


BACKENDS = {
    "sklearn": train_sklearn_backend,
    "fasttext": train_fasttext_backend,
    "bert": train_bert_backend,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS.keys()),
        default="sklearn",
        help="Training backend (default: sklearn).",
    )
    parser.add_argument(
        "--examples",
        "--input",
        dest="examples",
        type=Path,
        default=DEFAULT_EXAMPLES_V2,
        help=(
            "Path to training data. Accepts the legacy dict format "
            "(``intent_examples_v2.json``) or the augmented JSONL "
            "(``intent_train_v2.jsonl``)."
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT_V2,
        help="Output path for the trained model pickle.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_V2,
        help="Output path for the JSON classification report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config = TrainingConfig(
        backend=args.backend,
        examples_path=args.examples,
        artifact_path=args.artifact,
        report_path=args.report,
    )

    logger.info("Training v2 intent classifier | backend=%s", config.backend)
    logger.info("Examples: %s", config.examples_path)

    trainer = BACKENDS[config.backend]
    report = trainer(config)

    logger.info(
        "Done. accuracy=%s macro_f1=%s artifact=%s report=%s",
        report.get("accuracy"),
        report.get("macro_f1"),
        config.artifact_path,
        config.report_path,
    )
    logger.debug("Full report:\n%s", json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

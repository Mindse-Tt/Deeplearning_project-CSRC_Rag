"""HuggingFace 多标签微调封装：处罚类型（PunishmentType）多标签分类训练。

本模块是「处罚类型预测」这一辅助任务的训练入口。我们基于 transformers 的
``Trainer`` + ``AutoModelForSequenceClassification``，以 ``multi_label_classification``
问题类型微调一个预训练编码器：一条样本可同时命中多个处罚类型标签（如「警告+罚款」）。

设计取舍：
* 把超参收敛进 ``HFFineTuneConfig`` 这个不可变配置对象，训练逻辑只读不改，便于复现实验。
* 重依赖（transformers/datasets/torch）延迟到函数内部 import，缺包时给出可操作的报错而非
  在导入期就让整个项目崩掉。
* 以 Micro-F1 作为选优指标（多标签场景下对样本级整体表现更敏感），按 epoch 评估并保存最优。
* 训练/验证/测试三套标签共用同一份标签词表，保证标签维度与索引在三个集合上完全一致。
"""
from __future__ import annotations

from dataclasses import dataclass

from csrc_rag.training.data import build_label_vocab, encode_multilabel


@dataclass(frozen=True)
class HFFineTuneConfig:
    # 微调超参集中配置（不可变）：底座模型名、截断长度、批大小、学习率、训练轮数。
    model_name: str
    max_length: int
    batch_size: int
    learning_rate: float
    num_train_epochs: int


def run_hf_multilabel_finetune(
    train_rows: list[dict],
    valid_rows: list[dict],
    test_rows: list[dict],
    config: HFFineTuneConfig,
    output_dir: str,
) -> dict:
    """端到端跑一次多标签微调：建词表→编码→微调→在测试集评估，返回指标与词表。"""
    # 重依赖延迟导入：未安装 transformers/datasets/torch 时给出明确的环境修复指引。
    try:
        import numpy as np  # noqa: F401
        from datasets import Dataset  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face fine-tuning dependencies are unavailable. Install transformers, datasets, and torch in a Python 3.11 environment first."
        ) from exc

    from csrc_rag.training.metrics import multilabel_metrics

    # 三套数据合并构建标签词表，确保标签索引在 train/valid/test 上完全对齐。
    label_vocab = build_label_vocab(train_rows + valid_rows + test_rows)
    num_labels = len(label_vocab)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # 多标签分类头：problem_type 决定用 BCEWithLogits 损失；
    # ignore_mismatched_sizes 允许复用预训练权重的同时重建一个新尺寸的分类头。
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    def to_dataset(rows: list[dict]) -> Dataset:
        # 把原始样本转为 HF Dataset：input_text 作输入，多标签 0/1 向量作 labels。
        labels = encode_multilabel(rows, label_vocab).tolist()
        return Dataset.from_dict(
            {
                "text": [row["input_text"] for row in rows],
                "labels": labels,
            }
        )

    train_ds = to_dataset(train_rows)
    valid_ds = to_dataset(valid_rows)
    test_ds = to_dataset(test_rows)

    def preprocess(batch):
        # 仅做截断分词，不在此处 padding——交给 DataCollatorWithPadding 按 batch 动态补齐。
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

    train_ds = train_ds.map(preprocess, batched=True)
    valid_ds = valid_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)

    # 训练参数：按 epoch 评估并保存，以 Micro-F1 选最优模型；强制 CPU 训练以保证可复现，
    # save_safetensors=False 是为了规避部分环境下分类头权重落盘的兼容性问题。
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_Micro-F1",
        greater_is_better=True,
        report_to="none",
        use_cpu=True,
        save_safetensors=False,
    )

    def compute_metrics(eval_pred):
        import numpy as np

        # 多标签：对每个标签独立做 sigmoid，再以 0.5 为阈值二值化，逐标签判定是否命中。
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.float32)
        return multilabel_metrics(labels, preds)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # 训练结束后用（load_best_model_at_end 已回载的）最优模型在测试集上做最终评估。
    test_metrics = trainer.evaluate(test_ds)
    return {
        "label_vocab": label_vocab,
        "test_metrics": test_metrics,
        "model_name": config.model_name,
    }

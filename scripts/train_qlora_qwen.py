"""Qwen2.5-1.5B-Instruct 在 CSRC 处罚案例 RAG 问答数据上的 QLoRA 主训练脚本。

本脚本是我们微调流水线的底层训练入口：读取 Alpaca→chat 转换后的 JSONL 数据，
以 4-bit NF4 量化加载基座模型，挂载 LoRA 适配器，按配置文件中的超参做监督微调，
最终只保存轻量的 LoRA 适配器权重（而非整模），并落盘一份可复现 manifest。

运行环境（本项目实测）：
    - 本地 RTX 2060 SUPER (8GB)：4-bit 量化 + fp16 + 梯度检查点，是默认主力环境
    - Colab T4 / Kaggle P100 (16GB)：免费 GPU 备选
    - AutoDL A5000 / A100：算力不足时的付费兜底

输入：
    --train / --val 两个 JSONL，每行含 ``messages``（chat 三段式）与 ``source``
    （单字母 A..H 类别标记，供消融过滤）。
输出：
    --output 目录下的 LoRA 适配器、tokenizer 以及 train_manifest.json。

无 GPU 时可用 ``--debug`` 跑冒烟测试（0.5B 基座、1 epoch、200 条样本），
仅用于验证数据管线是否打通，不依赖 bitsandbytes。

用法::

    python scripts/train_qlora_qwen.py \\
        --config configs/qlora_config.json \\
        --train data/train/rag_qa_train.jsonl \\
        --val   data/train/rag_qa_val.jsonl \\
        --output artifacts/models/qwen_lora_csrc

消融实验（控制参与训练的数据类别，用于论文对照组）::

    python scripts/train_qlora_qwen.py --config configs/qlora_config.json \\
        --ablation v1     # 仅 A+B+C（基础三类模板）
    python scripts/train_qlora_qwen.py --config configs/qlora_config.json \\
        --ablation v2     # A+B+C+D+E+F（加入趋势/拒答/问候）
    python scripts/train_qlora_qwen.py --config configs/qlora_config.json \\
        --ablation v3     # 完整八类（默认）

依赖：transformers, peft, bitsandbytes, accelerate, trl, datasets。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------


# 我们把 configs/qlora_config.json 整个读进来，用只读 dataclass 包一层，
# 通过属性按区块（meta/lora/training/quantization 等）取值，避免散落的字典硬编码。
@dataclass(frozen=True)
class QLoRAConfig:
    raw: dict[str, Any]

    @property
    def base_model(self) -> str:
        return self.raw["meta"]["base_model"]

    @property
    def fallback_base_model(self) -> str:
        return self.raw["meta"]["fallback_base_model"]

    @property
    def lora(self) -> dict[str, Any]:
        return self.raw["lora"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def quantization(self) -> dict[str, Any]:
        return self.raw["quantization"]

    @property
    def early_stopping(self) -> dict[str, Any]:
        return self.raw["early_stopping"]

    @property
    def ablation_classes(self, version: str = "v3_full") -> list[str]:
        return self.raw["ablation"][version]


def load_config(path: Path) -> QLoRAConfig:
    with path.open("r", encoding="utf-8") as fp:
        return QLoRAConfig(raw=json.load(fp))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("missing data file: %s", path)
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def filter_by_ablation(samples: list[dict[str, Any]], allowed_sources: list[str]) -> list[dict[str, Any]]:
    # 消融过滤：只保留 source（类别字母）落在本轮允许集合内的样本，
    # 从而在不改数据文件的前提下切换 v1/v2/v3 三种数据配比。
    return [s for s in samples if s.get("source") in allowed_sources]


def format_chat_prompt(sample: dict[str, Any], tokenizer: Any) -> str:
    """Convert messages list into a single training string via chat template."""
    return tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False
    )


# ---------------------------------------------------------------------------
# Model setup (import-guarded so CPU-less debug runs don't crash on import)
# ---------------------------------------------------------------------------


def build_model_and_tokenizer(cfg: QLoRAConfig, debug: bool = False) -> tuple[Any, Any]:
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    # Policy: debug=True just shrinks the dataset (see main()), but we
    # keep the full 4-bit quantisation path so the smoke test actually
    # exercises the same memory profile as the main run. On GPUs with
    # <6 GB free headroom a non-quantised 0.5B blows up anyway because
    # activations dominate.
    model_name = cfg.base_model
    logger.info("loading base model: %s (debug=%s)", model_name, debug)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 只有在有 CUDA 时才走 4-bit 量化（QLoRA 的 Q）：用 NF4 + 双重量化把
    # 1.5B 基座压进 8GB 显存；compute_dtype 由配置给定（2060S 上为 fp16）。
    # CPU 调试路径不量化，避免 bitsandbytes 在无 GPU 时报错。
    bnb_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization["load_in_4bit"],
            bnb_4bit_quant_type=cfg.quantization["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg.quantization["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.quantization["bnb_4bit_compute_dtype"]
            ),
        )
        bnb_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        **bnb_kwargs,
    )

    if torch.cuda.is_available():
        from peft import prepare_model_for_kbit_training  # type: ignore

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.training["gradient_checkpointing"]
        )
    else:
        # CPU-only debug path: input_require_grads is needed for LoRA
        # to produce a connected graph under gradient checkpointing.
        if cfg.training.get("gradient_checkpointing"):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

    return model, tokenizer


def attach_lora(model: Any, cfg: QLoRAConfig) -> Any:
    # 挂载 LoRA 适配器（QLoRA 的 LoRA）：配置里取 r=16 / α=32（α=2r 的常用经验比，
    # 放大低秩增量的等效学习率），target_modules 覆盖注意力与 MLP 的投影层。
    # 冻结基座、只训练这部分低秩矩阵，可训练参数仅占全模型的极小比例。
    from peft import LoraConfig, get_peft_model  # type: ignore

    lora_config = LoraConfig(
        r=cfg.lora["r"],
        lora_alpha=cfg.lora["lora_alpha"],
        lora_dropout=cfg.lora["lora_dropout"],
        bias=cfg.lora["bias"],
        task_type=cfg.lora["task_type"],
        target_modules=cfg.lora["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def train(
    cfg: QLoRAConfig,
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    output_dir: Path,
    debug: bool = False,
) -> None:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers import (  # type: ignore
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    model, tokenizer = build_model_and_tokenizer(cfg, debug=debug)
    model = attach_lora(model, cfg)

    max_seq_length = cfg.training["max_seq_length"]

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        # 先用 chat 模板把 messages 拼成单条训练文本，再分词。
        # 这里采用全序列自回归监督（labels = input_ids 的拷贝），即对整段
        # 文本计算语言模型损失；padding 留给 collator 做动态批内补齐。
        text = format_chat_prompt(row, tokenizer)
        enc = tokenizer(
            text,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_ds = Dataset.from_list(train_samples).map(encode, remove_columns=["messages"])
    val_ds = Dataset.from_list(val_samples).map(encode, remove_columns=["messages"]) if val_samples else None

    # 训练超参：受限于 2060S 的 8GB 显存，单卡 batch 取小值（如 2），
    # 配合 gradient_accumulation_steps（如 8）把等效 batch 放大到 16，
    # 在显存可控的前提下逼近大 batch 的梯度稳定性。
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.training["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg.training["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg.training["gradient_accumulation_steps"],
        learning_rate=cfg.training["learning_rate"],
        num_train_epochs=1 if debug else cfg.training["num_train_epochs"],
        warmup_ratio=cfg.training["warmup_ratio"],
        lr_scheduler_type=cfg.training["lr_scheduler_type"],
        weight_decay=cfg.training["weight_decay"],
        optim=cfg.training["optim"] if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=cfg.training["gradient_checkpointing"],
        logging_steps=cfg.training["logging_steps"],
        eval_strategy=cfg.training["eval_strategy"] if val_ds else "no",
        eval_steps=cfg.training["eval_steps"],
        save_strategy=cfg.training["save_strategy"],
        save_steps=cfg.training["save_steps"],
        save_total_limit=cfg.training["save_total_limit"],
        load_best_model_at_end=cfg.training["load_best_model_at_end"] and val_ds is not None,
        metric_for_best_model=cfg.training["metric_for_best_model"],
        greater_is_better=cfg.training["greater_is_better"],
        # 混合精度自适应：RTX 2060S 是 Turing 架构，硬件不支持 bf16，
        # 因此仅在 GPU 确实支持 bf16 时才启用 bf16，否则回落到 fp16。
        # 这样同一份配置在 2060S（fp16）和 Ampere/A100（bf16）上都能直接跑。
        bf16=bool(cfg.training.get("bf16", False)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=bool(cfg.training.get("fp16", True)) and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=cfg.training["report_to"],
        seed=cfg.training["seed"],
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # 有验证集时启用早停：以验证指标连续若干次（patience）无 min_delta 改善
    # 即提前终止，避免在小数据上过拟合，也省算力。
    callbacks = []
    if val_ds is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping["patience"],
                early_stopping_threshold=cfg.early_stopping["min_delta"],
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("training done. adapter saved to %s", output_dir)

    # 落盘训练 manifest：记录基座、配置路径、训练/验证样本数与随机种子，
    # 用于复现实验与论文中标注的训练规模。
    manifest = {
        "base_model": cfg.base_model,
        "config_path": str(Path("configs/qlora_config.json").resolve()),
        "train_n": len(train_samples),
        "val_n": len(val_samples),
        "seed": cfg.training["seed"],
    }
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora_config.json"))
    parser.add_argument("--train", type=Path, default=Path("data/train/rag_qa_train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/train/rag_qa_val.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/models/qwen_lora_csrc")
    )
    parser.add_argument(
        "--ablation",
        choices=["v1", "v2", "v3"],
        default="v3",
        help="v1: A+B+C, v2: +D+E+F, v3: full 8 classes",
    )
    parser.add_argument("--debug", action="store_true", help="CPU debug on 0.5B base")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _ablation_key(flag: str) -> str:
    return {"v1": "v1_template_only", "v2": "v2_with_refuse", "v3": "v3_full"}[flag]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s %(name)s] %(message)s",
    )

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 设置 HF 镜像与关闭 tokenizers 并行，规避国内下载与多进程告警。
    cfg = load_config(args.config)
    # 根据 --ablation 取出本轮允许的数据类别集合，再据此过滤训练/验证集。
    allowed = cfg.raw["ablation"][_ablation_key(args.ablation)]

    train_all = load_jsonl(args.train)
    val_all = load_jsonl(args.val)
    train_filtered = filter_by_ablation(train_all, allowed)
    val_filtered = filter_by_ablation(val_all, allowed)

    logger.info(
        "ablation=%s train=%d (of %d) val=%d (of %d)",
        args.ablation,
        len(train_filtered),
        len(train_all),
        len(val_filtered),
        len(val_all),
    )

    if args.debug:
        train_filtered = train_filtered[:200]
        val_filtered = val_filtered[:40]

    if not train_filtered:
        logger.error("no training samples after filtering - did you run build_rag_qa_train.py?")
        return 1

    output_dir = args.output
    if args.ablation != "v3":
        output_dir = output_dir.with_name(output_dir.name + f"_{args.ablation}")

    train(cfg, train_filtered, val_filtered, output_dir, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

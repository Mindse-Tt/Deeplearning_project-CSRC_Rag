"""运行 2模型 × 3训练方式 = 6 组对比实验

模型:
  M1: Qwen/Qwen2.5-0.5B-Instruct (494M, 阿里)
  M2: bigscience/bloom-560m (560M, BigScience)

训练方式:
  T1: QLoRA (4-bit NF4 + LoRA r=16)
  T2: LoRA (fp16 + LoRA r=16, 不量化)
  T3: Full Fine-tune (全参数, fp16)

用法:
    python scripts/run_model_comparison.py --experiment M1_T1
    python scripts/run_model_comparison.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

LOGGER = logging.getLogger("model_comparison")

EXPERIMENTS = {
    "M1_T1": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "method": "qlora",
        "desc": "Qwen-0.5B + QLoRA",
    },
    "M1_T2": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "method": "lora",
        "desc": "Qwen-0.5B + LoRA (fp16)",
    },
    "M1_T3": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "method": "full",
        "desc": "Qwen-0.5B + Full Fine-tune",
    },
    "M2_T1": {
        "model": "bigscience/bloom-560m",
        "method": "qlora",
        "desc": "Bloom-560M + QLoRA",
    },
    "M2_T2": {
        "model": "bigscience/bloom-560m",
        "method": "lora",
        "desc": "Bloom-560M + LoRA (fp16)",
    },
    "M2_T3": {
        "model": "bigscience/bloom-560m",
        "method": "full",
        "desc": "Bloom-560M + Full Fine-tune",
    },
}


def train_experiment(exp_id: str, exp: dict, train_path: Path, val_path: Path, output_dir: Path) -> dict:
    """Run a single training experiment and return metrics."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    LOGGER.info(f"=== {exp_id}: {exp['desc']} ===")
    model_name = exp["model"]
    method = exp["method"]

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Load data
    with train_path.open(encoding="utf-8") as f:
        train_data = [json.loads(l) for l in f if l.strip()]
    with val_path.open(encoding="utf-8") as f:
        val_data = [json.loads(l) for l in f if l.strip()]

    # Subsample for speed (use 2000 train for comparison fairness)
    import random
    random.seed(42)
    if len(train_data) > 2000:
        train_data = random.sample(train_data, 2000)
    if len(val_data) > 200:
        val_data = random.sample(val_data, 200)

    LOGGER.info(f"  Data: {len(train_data)} train, {len(val_data)} val")

    # Load system prompt
    config_path = PROJECT_ROOT / "configs" / "qlora_config.json"
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    system_prompt = cfg["data"]["system_prompt"]

    # Convert to messages format for training
    def to_messages(row):
        user_text = row.get("instruction", "")
        if row.get("input"):
            user_text = f"{user_text}\n\n{row['input']}"
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": row.get("output", "")},
            ]
        }

    train_msgs = [to_messages(r) for r in train_data]
    val_msgs = [to_messages(r) for r in val_data]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    def tokenize(examples):
        texts = []
        for msg in examples:
            text = tokenizer.apply_chat_template(
                msg["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        encodings = tokenizer(
            texts, truncation=True, max_length=1024, padding="max_length"
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    # Create datasets - handle models with/without chat_template
    def format_text(messages):
        """Format messages into text, using chat_template if available, else simple concat."""
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except (ValueError, AttributeError):
            # Fallback for models without chat_template (e.g., Bloom)
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"[系统] {content}")
                elif role == "user":
                    parts.append(f"[用户] {content}")
                elif role == "assistant":
                    parts.append(f"[助手] {content}")
            return "\n".join(parts) + tokenizer.eos_token

    train_texts = [format_text(m["messages"]) for m in train_msgs]
    val_texts = [format_text(m["messages"]) for m in val_msgs]

    train_enc = tokenizer(train_texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

    train_enc["labels"] = train_enc["input_ids"].clone()
    val_enc["labels"] = val_enc["input_ids"].clone()

    train_dataset = Dataset.from_dict({k: v.tolist() for k, v in train_enc.items()})
    val_dataset = Dataset.from_dict({k: v.tolist() for k, v in val_enc.items()})

    # Load model based on method
    start_time = time.time()

    if method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] if "qwen" in model_name.lower()
            else ["query_key_value"],  # bloom uses different names
        )
        model = get_peft_model(model, lora_config)

    elif method == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] if "qwen" in model_name.lower()
            else ["query_key_value"],
        )
        model = get_peft_model(model, lora_config)

    elif method == "full":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
        # Enable gradient checkpointing for memory
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Training args
    exp_output = output_dir / exp_id
    exp_output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(exp_output),
        num_train_epochs=2,
        per_device_train_batch_size=2 if method == "qlora" else 1,
        per_device_eval_batch_size=2 if method == "qlora" else 1,
        gradient_accumulation_steps=8 if method == "qlora" else 16,
        learning_rate=2e-4 if method != "full" else 5e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=(method == "qlora"),  # Only use fp16 with quantized models
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    try:
        train_result = trainer.train()
        train_time = time.time() - start_time
        eval_result = trainer.evaluate()

        result = {
            "exp_id": exp_id,
            "desc": exp["desc"],
            "model": model_name,
            "method": method,
            "train_loss": round(train_result.metrics["train_loss"], 4),
            "eval_loss": round(eval_result["eval_loss"], 4),
            "train_time_s": round(train_time, 1),
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(trainable / total * 100, 2),
            "status": "success",
        }

        # Save adapter/model
        if method in ("qlora", "lora"):
            model.save_pretrained(str(exp_output / "adapter"))
        else:
            model.save_pretrained(str(exp_output / "model"))

        LOGGER.info(f"  Result: train_loss={result['train_loss']}, eval_loss={result['eval_loss']}, time={result['train_time_s']}s")

    except Exception as e:
        train_time = time.time() - start_time
        result = {
            "exp_id": exp_id,
            "desc": exp["desc"],
            "model": model_name,
            "method": method,
            "status": f"FAILED: {str(e)[:200]}",
            "train_time_s": round(train_time, 1),
        }
        LOGGER.error(f"  FAILED: {e}")

    # Clean up GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()), help="Run single experiment")
    parser.add_argument("--all", action="store_true", help="Run all 6 experiments")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "reports" / "model_comparison.json")
    args = parser.parse_args()

    train_path = PROJECT_ROOT / "data" / "processed" / "rag_qa_train.jsonl"
    val_path = PROJECT_ROOT / "data" / "processed" / "rag_qa_val.jsonl"
    output_dir = PROJECT_ROOT / "artifacts" / "experiments"

    if args.experiment:
        exps_to_run = {args.experiment: EXPERIMENTS[args.experiment]}
    elif args.all:
        exps_to_run = EXPERIMENTS
    else:
        parser.error("Specify --experiment or --all")
        return 1

    results = []
    for exp_id, exp in exps_to_run.items():
        result = train_experiment(exp_id, exp, train_path, val_path, output_dir)
        results.append(result)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Exp':<8} {'Description':<30} {'Train Loss':<12} {'Eval Loss':<12} {'Time':<10} {'Status'}")
    print("-" * 70)
    for r in results:
        print(f"{r['exp_id']:<8} {r['desc']:<30} {r.get('train_loss','N/A'):<12} {r.get('eval_loss','N/A'):<12} {r.get('train_time_s','N/A'):<10} {r['status']}")

    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

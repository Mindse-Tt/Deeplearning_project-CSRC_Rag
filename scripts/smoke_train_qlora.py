"""M1 smoke test: Qwen QLoRA fp16 on RTX 2060 SUPER 8GB.

Usage:
  python scripts/smoke_train_qlora.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --output_dir artifacts/smoke_qwen05_lora
  python scripts/smoke_train_qlora.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --output_dir artifacts/smoke_qwen15_lora --grad_accum 8

Records max GPU memory, writes metrics.json next to output_dir.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

# Prefer hf-mirror before anything imports transformers
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_DATA = ROOT / "data" / "train_smoke" / "smoke_qa.jsonl"


def format_prompt(instr: str, inp: str, out: str) -> str:
    # Qwen2.5 Instruct simple chat template (raw; we tokenize ourselves)
    if inp:
        user = f"{instr}\n{inp}"
    else:
        user = instr
    return (
        "<|im_start|>system\n你是证监会处罚案例分析助手。<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{out}<|im_end|>"
    )


def build_dataset(tokenizer, max_len: int = 384):
    ds = load_dataset("json", data_files=str(SMOKE_DATA), split="train")

    def _tok(ex):
        text = format_prompt(ex["instruction"], ex.get("input", ""), ex["output"])
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(_tok, remove_columns=ds.column_names)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=384)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    result = {
        "model": args.model,
        "output_dir": str(out_dir),
        "max_steps": args.max_steps,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "max_len": args.max_len,
        "status": "started",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }

    try:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print(f"[smoke] loading tokenizer: {args.model}")
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        print(f"[smoke] loading model (4bit nf4, fp16 compute): {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_cfg,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        print(f"[smoke] building dataset from {SMOKE_DATA}")
        train_ds = build_dataset(tok, max_len=args.max_len)
        print(f"[smoke] dataset size = {len(train_ds)}")

        collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

        targs = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=2e-4,
            num_train_epochs=1,
            fp16=True,
            bf16=False,
            max_steps=args.max_steps,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            warmup_steps=2,
            lr_scheduler_type="linear",
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            data_collator=collator,
        )

        print("[smoke] starting training …")
        train_out = trainer.train()

        # Collect loss history
        loss_hist = []
        for rec in trainer.state.log_history:
            if "loss" in rec:
                loss_hist.append({"step": rec.get("step"), "loss": rec["loss"]})

        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

        # Save LoRA adapter (few MB)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))

        result.update({
            "status": "ok",
            "elapsed_sec": round(time.time() - t0, 1),
            "train_runtime_sec": round(train_out.metrics.get("train_runtime", 0), 1),
            "train_samples_per_sec": round(train_out.metrics.get("train_samples_per_second", 0), 3),
            "peak_mem_allocated_gb": round(peak_alloc, 3),
            "peak_mem_reserved_gb": round(peak_reserved, 3),
            "loss_history": loss_hist,
            "final_loss": train_out.metrics.get("train_loss"),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
        print(json.dumps({k: v for k, v in result.items() if k != "loss_history"}, ensure_ascii=False, indent=2))

    except Exception as e:
        result.update({
            "status": "error",
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc(),
            "peak_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else None,
            "elapsed_sec": round(time.time() - t0, 1),
        })
        print(f"[smoke] ERROR: {e}")
        traceback.print_exc()

    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[smoke] metrics written to {metrics_path}")


if __name__ == "__main__":
    main()

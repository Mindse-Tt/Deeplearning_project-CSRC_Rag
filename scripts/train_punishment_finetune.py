from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.settings import CONFIG_DIR, PROCESSED_DIR
from csrc_rag.training.data import load_party_samples, time_split
from csrc_rag.training.hf_finetune import HFFineTuneConfig, run_hf_multilabel_finetune
from csrc_rag.utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for punishment-type multilabel classification.")
    parser.add_argument("--model-name", default=None, help="Override the transformer model name")
    parser.add_argument("--output-dir", default="artifacts/hf_finetune", help="Directory for checkpoints and outputs")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional smoke-test cap for training rows")
    parser.add_argument("--max-valid-samples", type=int, default=None, help="Optional smoke-test cap for validation rows")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional smoke-test cap for test rows")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--num-train-epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--max-length", type=int, default=None, help="Override tokenizer max length")
    args = parser.parse_args()

    model_config = read_json(CONFIG_DIR / "models.json")["fine_tuning"]
    rows = load_party_samples(str(PROCESSED_DIR / "party_samples.jsonl"))
    split = time_split(rows)
    train_rows = split.train[: args.max_train_samples] if args.max_train_samples else split.train
    valid_rows = split.valid[: args.max_valid_samples] if args.max_valid_samples else split.valid
    test_rows = split.test[: args.max_test_samples] if args.max_test_samples else split.test

    config = HFFineTuneConfig(
        model_name=args.model_name or model_config["transformer_model"],
        max_length=args.max_length or model_config["max_length"],
        batch_size=args.batch_size or model_config["batch_size"],
        learning_rate=float(model_config["learning_rate"]),
        num_train_epochs=args.num_train_epochs or int(model_config["num_train_epochs"]),
    )
    result = run_hf_multilabel_finetune(train_rows, valid_rows, test_rows, config=config, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

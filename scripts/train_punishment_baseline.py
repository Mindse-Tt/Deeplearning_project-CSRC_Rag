from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.settings import PROCESSED_DIR
from csrc_rag.training.data import load_party_samples, time_split
from csrc_rag.training.sklearn_baseline import train_tfidf_logreg


def main() -> None:
    rows = load_party_samples(str(PROCESSED_DIR / "party_samples.jsonl"))
    split = time_split(rows)
    output = train_tfidf_logreg(split.train, split.valid, split.test)
    payload = {
        "train_size": len(split.train),
        "valid_size": len(split.valid),
        "test_size": len(split.test),
        "metrics": output.metrics,
        "label_vocab_size": len(output.label_vocab),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.retrieval.dense import SentenceTransformerDenseEncoder, SvdTfidfDenseEncoder
from csrc_rag.settings import CONFIG_DIR, PROCESSED_DIR
from csrc_rag.utils import read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense retrieval assets.")
    parser.add_argument("--backend", choices=["svd_tfidf", "sentence_transformer"], default=None)
    args = parser.parse_args()

    model_config = read_json(CONFIG_DIR / "models.json")
    dense_config = model_config["dense_retrieval"]
    backend = args.backend or dense_config["backend"]

    chunks = read_jsonl(PROCESSED_DIR / "event_chunks.jsonl")
    doc_ids = [row["chunk_id"] for row in chunks]
    texts = [row["retrieval_text"] for row in chunks]

    if backend == "svd_tfidf":
        params = dense_config["svd_tfidf"]
        encoder = SvdTfidfDenseEncoder(
            max_features=params["max_features"],
            ngram_range=tuple(params["ngram_range"]),
            n_components=params["n_components"],
        )
    else:
        encoder = SentenceTransformerDenseEncoder(dense_config["sentence_transformer_model"])

    encoder.fit(doc_ids, texts)

    summary = {
        "backend": backend,
        "documents": len(doc_ids),
        "note": "当前索引构建在内存中完成。后续可扩展为持久化 embedding 或向量库。",
    }
    write_json(PROCESSED_DIR / f"dense_index_summary_{backend}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

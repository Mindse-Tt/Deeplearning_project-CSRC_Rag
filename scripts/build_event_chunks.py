from __future__ import annotations

import json

from csrc_rag.data.builders import write_jsonl
from csrc_rag.retrieval.chunking import build_event_chunks
from csrc_rag.settings import CONFIG_DIR, PROCESSED_DIR


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    event_corpus = load_jsonl(PROCESSED_DIR / "event_corpus.jsonl")
    config = json.loads((CONFIG_DIR / "retrieval.json").read_text(encoding="utf-8"))
    rows = build_event_chunks(
        event_documents=event_corpus,
        chunk_size=config["chunk_size"],
        overlap=config["chunk_overlap"],
    )
    output = PROCESSED_DIR / "event_chunks.jsonl"
    write_jsonl(output, rows)
    print("已生成:", output)
    print("chunk 数量:", len(rows))


if __name__ == "__main__":
    main()


from __future__ import annotations

import json

from csrc_rag.data.builders import build_event_corpus, build_party_samples, write_jsonl
from csrc_rag.data.excel_loader import iter_records
from csrc_rag.settings import PROCESSED_DIR, RAW_WORKBOOK, ensure_output_dirs


def main() -> None:
    ensure_output_dirs()
    records = iter_records(RAW_WORKBOOK)
    event_corpus = build_event_corpus(records)
    party_samples = build_party_samples(records)

    event_path = PROCESSED_DIR / "event_corpus.jsonl"
    party_path = PROCESSED_DIR / "party_samples.jsonl"
    summary_path = PROCESSED_DIR / "build_summary.json"

    write_jsonl(event_path, event_corpus)
    write_jsonl(party_path, party_samples)
    summary_path.write_text(
        json.dumps(
            {
                "raw_rows": len(records),
                "event_documents": len(event_corpus),
                "party_samples": len(party_samples),
                "event_corpus_path": str(event_path),
                "party_samples_path": str(party_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("已生成:")
    print(" ", event_path)
    print(" ", party_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()

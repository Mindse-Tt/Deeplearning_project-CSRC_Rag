"""Collect 1-2 qualitative case studies where rerank changes the top-5.

Used to populate the M2 retrieval report §Sample Cases. Writes a JSON file
with query + hybrid top-5 + hybrid+rerank top-5 + event titles.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_CACHE = str(PROJECT_ROOT / "artifacts" / "models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _CACHE)
os.environ.setdefault("HF_HOME", _CACHE)
os.environ.setdefault("HF_HUB_CACHE", _CACHE)

from csrc_rag.retrieval.engine import RetrievalEngine  # noqa: E402

QUERIES = [
    "2023年上市公司财务造假的处罚案例",
    "基金公司从业人员私下买卖股票被罚",
    "内幕交易被证监会罚款",
    "信息披露违规的上市公司",
    "操纵市场案件的处罚决定",
]


def row_to_public(event: dict, score: float) -> dict:
    return {
        "event_id": event["event_id"],
        "title": event.get("title"),
        "declare_date": event.get("declare_date"),
        "promulgator": event.get("promulgator"),
        "punishment_types": event.get("punishment_types", []),
        "score": round(score, 4),
    }


def main() -> None:
    engine = RetrievalEngine(retrieval_mode="hybrid", rerank_enabled=False)
    engine._get_reranker()  # pre-load

    cases = []
    for q in QUERIES:
        engine.rerank_enabled = False
        r0 = engine.search(q, forced_intent="case_retrieval")
        engine.rerank_enabled = True
        r1 = engine.search(q, forced_intent="case_retrieval")
        cases.append(
            {
                "query": q,
                "hybrid_top5": [
                    {"event_id": e["event_id"], "title": e.get("title"), "score": e.get("score")}
                    for e in r0.events[:5]
                ],
                "rerank_top5": [
                    {"event_id": e["event_id"], "title": e.get("title"), "score": e.get("score")}
                    for e in r1.events[:5]
                ],
            }
        )

    out = PROJECT_ROOT / "docs" / "reports" / "m2_retrieval_cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": cases}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

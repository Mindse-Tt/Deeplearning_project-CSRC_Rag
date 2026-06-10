from __future__ import annotations

import json
from statistics import mean

from csrc_rag.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from csrc_rag.response.responder import TemplateResponder
from csrc_rag.retrieval.engine import RetrievalEngine
from csrc_rag.settings import PROCESSED_DIR


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_query(event: dict) -> str | None:
    activity = event.get("activity")
    if not activity:
        return None
    return activity[:160]


def main() -> None:
    events = load_jsonl(PROCESSED_DIR / "event_corpus.jsonl")
    engine = RetrievalEngine()
    engine.responder = TemplateResponder()

    recalls = []
    mrrs = []
    ndcgs = []
    tested = 0
    for event in events[:300]:
        query = build_query(event)
        if not query:
            continue
        response = engine.search(query, forced_intent="case_retrieval")
        ranked_event_ids = [item["event_id"] for item in response.events]
        gold = event["event_id"]
        recalls.append(recall_at_k(ranked_event_ids, gold, 5))
        mrrs.append(reciprocal_rank(ranked_event_ids, gold))
        ndcgs.append(ndcg_at_k(ranked_event_ids, gold, 10))
        tested += 1

    print(json.dumps(
        {
            "tested_queries": tested,
            "Recall@5": round(mean(recalls), 4) if recalls else 0.0,
            "MRR": round(mean(mrrs), 4) if mrrs else 0.0,
            "nDCG@10": round(mean(ndcgs), 4) if ndcgs else 0.0,
            "note": "这是第一阶段的自举检索 sanity check，验证系统能否找回源事件及其证据片段；后续还需要人工标注的跨案例相似性评测。",
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

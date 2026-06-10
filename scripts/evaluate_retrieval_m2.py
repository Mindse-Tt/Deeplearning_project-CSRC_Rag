"""M2 retrieval ablation: compare 4 retrieval backends on the sanity set.

Computes Recall@5 / MRR / nDCG@10 for:

    1. BM25-only
    2. Dense-only (bge-small-zh-v1.5)
    3. Hybrid (BM25 + Dense + RRF)
    4. Hybrid + Reranker (bge-reranker-v2-m3)

Two evaluation modes
--------------------
* **sanity (default)** — for each event we use ``event.activity`` as the
  query and the same ``event_id`` as the single gold label. Auto-bootstrap;
  numbers are optimistic but the *relative* improvement between backends
  is what matters.
* **gold (``--eval data/eval/gold_50.jsonl``)** — 50 human-curated
  case_retrieval / law_grounding / ... queries with one or more acceptable
  ``relevant_event_ids`` per row. Used for the M3 milestone target
  (Recall@5 ≥ 0.35 on BM25-only, ≥ 0.60 with Hybrid+Rerank).

Usage
-----
    # Sanity self-bootstrap
    python scripts/evaluate_retrieval_m2.py \\
        --limit 200 \\
        --output docs/reports/m2_retrieval_eval.json

    # M3 gold evaluation (markdown report)
    python scripts/evaluate_retrieval_m2.py \\
        --eval data/eval/gold_50.jsonl \\
        --output docs/reports/m3_retrieval_report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Route HuggingFace through the mirror and point cache to artifacts/models.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_CACHE = str(PROJECT_ROOT / "artifacts" / "models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _CACHE)
os.environ.setdefault("HF_HOME", _CACHE)
os.environ.setdefault("HF_HUB_CACHE", _CACHE)

from csrc_rag.evaluation.retrieval_metrics import (  # noqa: E402
    hit_at_k_multi,
    ndcg_at_k,
    ndcg_at_k_multi,
    recall_at_k,
    recall_at_k_multi,
    reciprocal_rank,
    reciprocal_rank_multi,
)
from csrc_rag.response.responder import TemplateResponder  # noqa: E402
from csrc_rag.retrieval.engine import RetrievalEngine  # noqa: E402
from csrc_rag.settings import PROCESSED_DIR  # noqa: E402

LOGGER = logging.getLogger("evaluate_retrieval_m2")


# Gold intents that go through retrieval. The remainder (out_of_scope,
# multi_turn_followup without relevant_event_ids) are skipped here because
# retrieval is not on their critical path.
_RETRIEVAL_INTENTS = frozenset(
    {"case_retrieval", "law_grounding", "sanction_recommendation", "trend_analysis"}
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_query(event: dict) -> str | None:
    activity = event.get("activity")
    if not activity:
        return None
    return activity[:160]


def evaluate(
    engine: RetrievalEngine,
    events: list[dict],
    *,
    label: str,
) -> dict:
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    tested = 0
    t0 = time.perf_counter()
    for i, event in enumerate(events):
        query = build_query(event)
        if not query:
            continue
        try:
            response = engine.search(query, forced_intent="case_retrieval")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("[%s] query %d failed: %s", label, i, exc)
            continue
        ranked_event_ids = [item["event_id"] for item in response.events]
        gold = event["event_id"]
        recalls.append(recall_at_k(ranked_event_ids, gold, 5))
        mrrs.append(reciprocal_rank(ranked_event_ids, gold))
        ndcgs.append(ndcg_at_k(ranked_event_ids, gold, 10))
        tested += 1

    elapsed = time.perf_counter() - t0
    result = {
        "label": label,
        "tested_queries": tested,
        "Recall@5": round(mean(recalls), 4) if recalls else 0.0,
        "MRR": round(mean(mrrs), 4) if mrrs else 0.0,
        "nDCG@10": round(mean(ndcgs), 4) if ndcgs else 0.0,
        "latency_total_s": round(elapsed, 2),
        "latency_per_query_ms": round(elapsed * 1000.0 / max(tested, 1), 1),
    }
    return result


def _filter_gold_rows(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for row in rows:
        if row.get("is_trap"):
            continue
        if row.get("intent") not in _RETRIEVAL_INTENTS:
            continue
        if not row.get("relevant_event_ids"):
            continue
        if not row.get("query"):
            continue
        kept.append(row)
    return kept


def evaluate_gold(
    engine: RetrievalEngine,
    gold_rows: list[dict],
    *,
    label: str,
) -> dict:
    """Evaluate against the human-curated gold set (multi-gold per question)."""
    recalls: list[float] = []
    hits: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    tested = 0
    t0 = time.perf_counter()
    for i, row in enumerate(gold_rows):
        query = row["query"]
        intent = row.get("intent") or "case_retrieval"
        try:
            response = engine.search(query, forced_intent=intent)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("[%s] row %s failed: %s", label, row.get("id"), exc)
            continue
        ranked_event_ids = [item["event_id"] for item in response.events]
        gold = [str(x) for x in row.get("relevant_event_ids", [])]
        recalls.append(recall_at_k_multi(ranked_event_ids, gold, 5))
        hits.append(hit_at_k_multi(ranked_event_ids, gold, 5))
        mrrs.append(reciprocal_rank_multi(ranked_event_ids, gold))
        ndcgs.append(ndcg_at_k_multi(ranked_event_ids, gold, 10))
        tested += 1

    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "tested_queries": tested,
        "Recall@5": round(mean(recalls), 4) if recalls else 0.0,
        "Hit@5": round(mean(hits), 4) if hits else 0.0,
        "MRR": round(mean(mrrs), 4) if mrrs else 0.0,
        "nDCG@10": round(mean(ndcgs), 4) if ndcgs else 0.0,
        "latency_total_s": round(elapsed, 2),
        "latency_per_query_ms": round(elapsed * 1000.0 / max(tested, 1), 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2/M3 retrieval ablation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help=(
            "Path to a gold .jsonl eval set with rows of the form "
            "{query, intent, relevant_event_ids, ...}. If omitted, the "
            "sanity self-bootstrap set is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "reports" / "m2_retrieval_eval.json",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["bm25", "dense", "hybrid", "rerank"],
        help="Skip specific conditions (useful for iterative debugging).",
    )
    return parser.parse_args()


def _render_markdown(
    title: str, config: dict, results: list[dict], extra_note: str | None = None
) -> str:
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"- 评测集: `{config.get('dataset')}`")
    lines.append(f"- 查询条数: {config.get('tested_total', '?')}")
    if extra_note:
        lines.append("")
        lines.append(extra_note)
    lines.append("")
    lines.append("## 4 档消融对比")
    lines.append("")
    cols = ["label", "tested_queries", "Recall@5", "Hit@5", "MRR", "nDCG@10", "latency_per_query_ms"]
    # Only include Hit@5 column if at least one result has it.
    if not any("Hit@5" in r for r in results):
        cols.remove("Hit@5")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines.append(header)
    lines.append(sep)
    for r in results:
        row = "| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |"
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

    use_gold = args.eval is not None
    if use_gold:
        raw_rows = load_jsonl(args.eval)
        gold_rows = _filter_gold_rows(raw_rows)
        LOGGER.info(
            "Loaded %d gold rows (kept %d retrieval-eligible) from %s",
            len(raw_rows),
            len(gold_rows),
            args.eval,
        )
        dataset_label = f"gold ({args.eval.name})"
    else:
        events = load_jsonl(PROCESSED_DIR / "event_corpus.jsonl")[: args.limit]
        LOGGER.info("Loaded %d events for sanity evaluation", len(events))
        dataset_label = "sanity_self_bootstrap (activity → event_id)"

    results: list[dict] = []

    def _run(engine: RetrievalEngine, label: str) -> None:
        if use_gold:
            results.append(evaluate_gold(engine, gold_rows, label=label))
        else:
            results.append(evaluate(engine, events, label=label))
        LOGGER.info("%s result: %s", label, results[-1])

    # 1. BM25-only
    if "bm25" not in args.skip:
        LOGGER.info("### 1/4 BM25-only")
        engine = RetrievalEngine(retrieval_mode="bm25", rerank_enabled=False)
        engine.responder = TemplateResponder()
        _run(engine, "bm25")
        del engine

    # 2. Dense-only (bge-small-zh)
    if "dense" not in args.skip:
        LOGGER.info("### 2/4 Dense-only (bge-small-zh)")
        engine = RetrievalEngine(retrieval_mode="dense", rerank_enabled=False)
        engine.responder = TemplateResponder()
        _run(engine, "dense_bge")
        del engine

    # 3. Hybrid (BM25 + Dense + RRF)
    hybrid_engine = None
    if "hybrid" not in args.skip or "rerank" not in args.skip:
        hybrid_engine = RetrievalEngine(
            retrieval_mode="hybrid", rerank_enabled=False
        )
        hybrid_engine.responder = TemplateResponder()

    if "hybrid" not in args.skip and hybrid_engine is not None:
        LOGGER.info("### 3/4 Hybrid (RRF)")
        _run(hybrid_engine, "hybrid_rrf")

    # 4. Hybrid + Reranker
    if "rerank" not in args.skip and hybrid_engine is not None:
        LOGGER.info("### 4/4 Hybrid + Reranker (bge-reranker-v2-m3)")
        hybrid_engine.rerank_enabled = True
        # Pre-load the reranker once.
        hybrid_engine._get_reranker()
        _run(hybrid_engine, "hybrid_rerank")

    config = {
        "limit": args.limit,
        "dataset": dataset_label,
        "skip": args.skip,
        "tested_total": max((r.get("tested_queries", 0) for r in results), default=0),
        "eval_path": str(args.eval) if use_gold else None,
    }
    output = {"config": config, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.suffix.lower() == ".md":
        md = _render_markdown(
            title="M3 检索升级评估报告",
            config=config,
            results=results,
            extra_note=(
                "> 配置：R2 软过滤 + R3 jieba tokenizer + R4 候选池 100。"
                " BM25 `k1=1.2,b=0.75`，RRF `k=60`。"
            ),
        )
        args.output.write_text(md, encoding="utf-8")
        # Always also drop a sibling JSON for machine-readability.
        json_out = args.output.with_suffix(".json")
        json_out.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("Wrote %s and %s", args.output, json_out)
    else:
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info("Wrote %s", args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

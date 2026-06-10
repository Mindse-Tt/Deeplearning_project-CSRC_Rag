"""Evaluate the L6 Trend Aggregator against gold_trend_30.

Metric suite
------------
* **Exact match rate** — count identical to expected (per bucket)
* **Approximate error rate (AER)** — mean relative error across buckets
* **Ranking accuracy** — for non-year facets, does our top-3 match expected top-3?
* **Peak year accuracy** — queries that require "哪一年最高", does the peak agree?

Usage::

    python scripts/evaluate_trend_aggregator.py \\
        --gold data/eval/gold_trend_30.jsonl \\
        --out  docs/reports/m4_2_trend_eval.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.orchestration.trend_aggregator import TrendAggregator  # noqa: E402

LOGGER = logging.getLogger("evaluate_trend")


def _relative_error(actual: int, expected: int) -> float:
    if expected == 0:
        return 0.0 if actual == 0 else 1.0
    return abs(actual - expected) / expected


def evaluate_row(agg: TrendAggregator, row: dict) -> dict:
    """Run aggregator on one row; compare to ``expected_aggregation``.

    Returns per-row metrics. Rows without ``expected_aggregation`` are
    skipped (legacy trend rows from gold_50).
    """
    expected = row.get("expected_aggregation")
    if not expected:
        return {"id": row["id"], "skipped": True, "reason": "no_expected"}

    facet = expected["facet"]
    window = tuple(expected["year_window"]) if expected.get("year_window") else None
    slot_filters = expected.get("slot_filters") or {}

    result = agg.aggregate(
        row["query"],
        facets=(facet,),
        year_window=window,
        slot_filters=slot_filters,
    )
    actual_buckets = {v.key: v.count for v in result.slices[0].values}
    expected_buckets = {b["key"]: b["count"] for b in expected["buckets"]}

    # Per-bucket metrics
    keys = sorted(set(actual_buckets) | set(expected_buckets))
    per_key = []
    errors: list[float] = []
    exact_hits = 0
    for k in keys:
        a = actual_buckets.get(k, 0)
        e = expected_buckets.get(k, 0)
        err = _relative_error(a, e)
        errors.append(err)
        if a == e:
            exact_hits += 1
        per_key.append({"key": k, "actual": a, "expected": e, "rel_err": round(err, 4)})

    # Ranking accuracy (top-3 for non-year facets)
    rank_acc: float | None = None
    if facet != "year" and len(expected_buckets) >= 3:
        a_top3 = tuple(sorted(actual_buckets, key=lambda k: -actual_buckets[k])[:3])
        e_top3 = tuple(sorted(expected_buckets, key=lambda k: -expected_buckets[k])[:3])
        rank_acc = 1.0 if a_top3 == e_top3 else (
            len(set(a_top3) & set(e_top3)) / 3.0
        )

    # Peak year accuracy
    peak_year_correct: bool | None = None
    if facet == "year":
        if actual_buckets and expected_buckets:
            a_peak = max(actual_buckets, key=lambda k: actual_buckets[k])
            e_peak = max(expected_buckets, key=lambda k: expected_buckets[k])
            peak_year_correct = (a_peak == e_peak)

    return {
        "id": row["id"],
        "query": row["query"],
        "facet": facet,
        "n_keys_actual": len(actual_buckets),
        "n_keys_expected": len(expected_buckets),
        "exact_bucket_hits": exact_hits,
        "exact_bucket_rate": round(exact_hits / max(len(keys), 1), 4),
        "mean_rel_err": round(mean(errors), 4) if errors else 0.0,
        "ranking_accuracy": rank_acc,
        "peak_year_correct": peak_year_correct,
        "per_key": per_key,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "gold_trend_30.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "reports" / "m4_2_trend_eval.md",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    agg = TrendAggregator.from_jsonl(
        PROJECT_ROOT / "data" / "processed" / "event_corpus.jsonl"
    )
    rows = [
        json.loads(line)
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results: list[dict] = []
    skipped = 0
    for row in rows:
        r = evaluate_row(agg, row)
        if r.get("skipped"):
            skipped += 1
            continue
        results.append(r)

    # Aggregate metrics
    if results:
        macro_exact_rate = mean(r["exact_bucket_rate"] for r in results)
        macro_rel_err = mean(r["mean_rel_err"] for r in results)
        ranking_results = [r["ranking_accuracy"] for r in results if r["ranking_accuracy"] is not None]
        macro_rank_acc = mean(ranking_results) if ranking_results else None
        peak_results = [r["peak_year_correct"] for r in results if r["peak_year_correct"] is not None]
        peak_acc = sum(peak_results) / len(peak_results) if peak_results else None
    else:
        macro_exact_rate = macro_rel_err = 0.0
        macro_rank_acc = peak_acc = None

    report: dict = {
        "evaluator": "TrendAggregator vs gold_trend_30",
        "n_eligible": len(results),
        "n_skipped": skipped,
        "macro_exact_bucket_rate": round(macro_exact_rate, 4),
        "macro_mean_rel_err": round(macro_rel_err, 4),
        "macro_ranking_accuracy": round(macro_rank_acc, 4) if macro_rank_acc is not None else None,
        "peak_year_accuracy": round(peak_acc, 4) if peak_acc is not None else None,
        "per_row": results,
    }

    # Render markdown
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# M4.2 Trend Aggregator 定量评估",
        "",
        f"- 评测集: `{args.gold.relative_to(PROJECT_ROOT)}`",
        f"- 参评/跳过: {len(results)} / {skipped}",
        f"- **Exact bucket rate**: {report['macro_exact_bucket_rate']}",
        f"- **Mean relative error**: {report['macro_mean_rel_err']}",
        f"- **Ranking accuracy (top-3)**: {report['macro_ranking_accuracy']}",
        f"- **Peak year accuracy**: {report['peak_year_accuracy']}",
        "",
        "## Per-row metrics",
        "",
        "| id | facet | exact | rel_err | rank_acc | peak_ok |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['facet']} | {r['exact_bucket_rate']} | "
            f"{r['mean_rel_err']} | "
            f"{r['ranking_accuracy'] if r['ranking_accuracy'] is not None else '-'} | "
            f"{r['peak_year_correct'] if r['peak_year_correct'] is not None else '-'} |"
        )

    args.out.write_text("\n".join(lines), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[PASS] wrote {args.out.relative_to(PROJECT_ROOT)}")
    print(f"[PASS] wrote {args.out.with_suffix('.json').relative_to(PROJECT_ROOT)}")
    print()
    print(
        f"summary: exact={report['macro_exact_bucket_rate']} "
        f"err={report['macro_mean_rel_err']} "
        f"rank={report['macro_ranking_accuracy']} "
        f"peak={report['peak_year_accuracy']}"
    )


if __name__ == "__main__":
    main()

"""Build +20 trend_analysis gold rows with expected_aggregation.

Usage::

    python scripts/build_trend_gold_extra.py

Reads the current ``data/eval/gold_100.jsonl`` (110 rows, 10 trend) and
appends 20 more ``trend_analysis`` rows with an ``expected_aggregation``
field — the ground-truth count(s) that the L6 aggregator should return
for that query. The expected counts are computed on the fly from
``data/processed/event_corpus.jsonl`` so the gold stays in sync with the
corpus even if it is rebuilt.

Each new row carries:

    {
      "id": "gold_111",
      "intent": "trend_analysis",
      "query": "...",
      "gold_answer_keypoints": [...],
      "relevant_event_ids": [],       # trend_analysis 不要求具体 EID
      "relevant_laws": [],
      "expected_slots": {...},
      "expected_aggregation": {
        "facet": "year" | "violation_type" | "punishment_type" | "agency",
        "year_window": [2022, 2024],
        "slot_filters": {"violation_type": ["内幕交易"]},
        "buckets": [
          {"key": "2022", "count": 13},
          {"key": "2023", "count": 20},
          {"key": "2024", "count": 41}
        ],
        "tolerance": "±15%"
      },
      "difficulty": "easy" | "medium" | "hard",
      "notes": "..."
    }

Output: ``data/eval/gold_trend_30.jsonl`` (original 10 + new 20 = 30
trend rows standalone) and ``data/eval/gold_130.jsonl`` (full set).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.orchestration.trend_aggregator import TrendAggregator  # noqa: E402


GOLD_100 = PROJECT_ROOT / "data" / "eval" / "gold_100.jsonl"
CORPUS = PROJECT_ROOT / "data" / "processed" / "event_corpus.jsonl"
OUT_TREND30 = PROJECT_ROOT / "data" / "eval" / "gold_trend_30.jsonl"
OUT_GOLD130 = PROJECT_ROOT / "data" / "eval" / "gold_130.jsonl"


# The 20 new trend queries cover the facets we support, in a mix of
# easy/medium/hard. We resolve the expected numbers at build time via
# the aggregator so they stay truthful against the current corpus.
NEW_QUERIES: list[dict] = [
    # --- year-trend, different violation types ---
    {
        "q": "2020-2024 年内幕交易案件数量逐年变化情况？",
        "slots": {"violation_type": ["内幕交易"]},
        "facet": "year",
        "window": [2020, 2024],
        "difficulty": "easy",
        "keypoints": ["必须给出 ≥ 3 个年份数字", "必须说明趋势方向（上升/下降/波动）"],
    },
    {
        "q": "虚构利润类案件 2019-2024 年数量演变？",
        "slots": {"violation_type": ["虚构利润"]},
        "facet": "year",
        "window": [2019, 2024],
        "difficulty": "easy",
        "keypoints": ["必须给出 ≥ 3 个年份数字"],
    },
    {
        "q": "推迟披露类违规 2021-2024 年数量分布？",
        "slots": {"violation_type": ["推迟披露"]},
        "facet": "year",
        "window": [2021, 2024],
        "difficulty": "easy",
        "keypoints": ["必须给出 4 个年份数字", "必须给出趋势判断"],
    },
    {
        "q": "重大遗漏类案件 2020-2024 年度数量变化？",
        "slots": {"violation_type": ["重大遗漏"]},
        "facet": "year",
        "window": [2020, 2024],
        "difficulty": "easy",
        "keypoints": ["必须给出 ≥ 3 个年份数字"],
    },
    {
        "q": "违规买卖股票类案件 2022-2024 年是否呈上升趋势？",
        "slots": {"violation_type": ["违规买卖股票"]},
        "facet": "year",
        "window": [2022, 2024],
        "difficulty": "easy",
        "keypoints": ["必须给出 3 个年份数字", "必须给出上升/下降结论"],
    },
    # --- punishment_type trend ---
    {
        "q": "没收非法所得类处罚 2020-2024 年应用情况？",
        "slots": {"punishment_type": ["没收非法所得"]},
        "facet": "year",
        "window": [2020, 2024],
        "difficulty": "medium",
        "keypoints": ["必须给出 ≥ 3 个年份数字", "必须说明趋势方向"],
    },
    {
        "q": "警告类处罚 2022-2024 年数量变化？",
        "slots": {"punishment_type": ["警告"]},
        "facet": "year",
        "window": [2022, 2024],
        "difficulty": "easy",
        "keypoints": ["必须给出 3 个年份数字"],
    },
    # --- full-window distribution queries ---
    {
        "q": "近 5 年各违规类型占比如何？",
        "slots": {},
        "facet": "violation_type",
        "window": [2021, 2025],
        "difficulty": "medium",
        "keypoints": ["必须给出 TOP-5 违规类型", "必须给出各类占比百分数"],
    },
    {
        "q": "2024 年所有行政处罚按处罚方式分布？",
        "slots": {},
        "facet": "punishment_type",
        "window": [2024, 2024],
        "difficulty": "easy",
        "keypoints": ["必须列出所有出现的处罚方式及其 count"],
    },
    {
        "q": "整个证监会数据库里哪些违规类型最常见？",
        "slots": {},
        "facet": "violation_type",
        "window": None,
        "difficulty": "easy",
        "keypoints": ["必须给出 TOP-5 违规类型按 count 降序", "必须给出具体数字"],
    },
    {
        "q": "全量数据里处罚方式的分布如何？",
        "slots": {},
        "facet": "punishment_type",
        "window": None,
        "difficulty": "easy",
        "keypoints": ["必须列出所有处罚方式", "必须给出 count 排序"],
    },
    # --- agency rankings ---
    {
        "q": "各监管机构在 2023-2024 年处罚案件数量排名？",
        "slots": {},
        "facet": "agency",
        "window": [2023, 2024],
        "difficulty": "medium",
        "keypoints": ["必须给出 TOP-3 机构名", "必须给出各机构 count"],
    },
    {
        "q": "2024 年哪个监管机构发出的处罚最多？",
        "slots": {},
        "facet": "agency",
        "window": [2024, 2024],
        "difficulty": "easy",
        "keypoints": ["必须点出 TOP-1 机构", "必须给出数字"],
    },
    # --- combined filter + facet ---
    {
        "q": "2024 年内幕交易案件按处罚方式分布？",
        "slots": {"violation_type": ["内幕交易"]},
        "facet": "punishment_type",
        "window": [2024, 2024],
        "difficulty": "medium",
        "keypoints": ["必须给出至少 3 种处罚方式及其 count"],
    },
    {
        "q": "2023-2024 年操纵股价案件按年度分布？",
        "slots": {"violation_type": ["操纵股价"]},
        "facet": "year",
        "window": [2023, 2024],
        "difficulty": "medium",
        "keypoints": ["必须给出两年数字"],
    },
    {
        "q": "2022-2024 年虚假记载案件按年度分布？",
        "slots": {"violation_type": ["虚假记载"]},
        "facet": "year",
        "window": [2022, 2024],
        "difficulty": "easy",
        "keypoints": ["必须给出 3 个年份数字"],
    },
    # --- long-horizon trends ---
    {
        "q": "2017-2024 年整体行政处罚数量逐年变化？",
        "slots": {},
        "facet": "year",
        "window": [2017, 2024],
        "difficulty": "medium",
        "keypoints": ["必须给出 ≥ 5 个年份的数字", "必须给出总体趋势判断"],
    },
    {
        "q": "2015-2024 年内幕交易案件历年数据？",
        "slots": {"violation_type": ["内幕交易"]},
        "facet": "year",
        "window": [2015, 2024],
        "difficulty": "hard",
        "keypoints": ["必须给出 ≥ 5 个年份数字", "必须说明长期趋势"],
    },
    # --- hard: combined facet detection on implicit year ---
    {
        "q": "近 3 年证监会市场禁入类处罚主要集中在哪些违规类型？",
        "slots": {"punishment_type": ["市场禁入"]},
        "facet": "violation_type",
        "window": [2023, 2025],
        "difficulty": "hard",
        "keypoints": ["必须点出 TOP-3 违规类型", "必须给出数字"],
    },
    {
        "q": "近 5 年哪些年度的违规买卖股票案件数量最高？",
        "slots": {"violation_type": ["违规买卖股票"]},
        "facet": "year",
        "window": [2021, 2025],
        "difficulty": "medium",
        "keypoints": ["必须给出各年份数字", "必须点出峰值年份"],
    },
]


def build_expected_aggregation(
    agg: TrendAggregator, item: dict
) -> dict:
    """Resolve ground-truth buckets from the aggregator."""
    window: tuple[int, int] | None = (
        tuple(item["window"]) if item.get("window") else None  # type: ignore[assignment]
    )
    slot_filters = item.get("slots") or {}
    result = agg.aggregate(
        item["q"],
        facets=(item["facet"],),
        year_window=window,
        slot_filters=slot_filters,
    )
    slice_ = result.slices[0]
    return {
        "facet": item["facet"],
        "year_window": list(window) if window else None,
        "slot_filters": slot_filters,
        "buckets": [
            {"key": v.key, "count": v.count, "share": v.share}
            for v in slice_.values
        ],
        "total_events": slice_.total,
        "tolerance": "±20%",
    }


def main() -> None:
    agg = TrendAggregator.from_jsonl(CORPUS)
    existing = [json.loads(l) for l in GOLD_100.read_text(encoding="utf-8").splitlines() if l.strip()]
    trend_existing = [r for r in existing if r.get("intent") == "trend_analysis"]

    new_rows: list[dict] = []
    for i, item in enumerate(NEW_QUERIES, start=1):
        gid = f"gold_{110 + i:03d}"
        expected = build_expected_aggregation(agg, item)
        slots_meta = {
            "aggregation": item["facet"],
            "year_range": (
                f"{item['window'][0]}-{item['window'][1]}"
                if item.get("window") else "all"
            ),
        }
        for k, v in (item.get("slots") or {}).items():
            slots_meta[k] = "/".join(str(x) for x in v)

        new_rows.append({
            "id": gid,
            "intent": "trend_analysis",
            "query": item["q"],
            "gold_answer_keypoints": item.get("keypoints", []),
            "relevant_event_ids": [],
            "relevant_laws": [],
            "expected_slots": slots_meta,
            "expected_aggregation": expected,
            "difficulty": item["difficulty"],
            "is_trap": False,
            "trap_reason": None,
            "notes": f"auto-generated from aggregator on {len(agg._corpus)} events",
        })

    # Write the 30-trend standalone file
    with OUT_TREND30.open("w", encoding="utf-8") as fh:
        for r in trend_existing:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in new_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write the full 130 gold file (original 110 + 20 new trend)
    with OUT_GOLD130.open("w", encoding="utf-8") as fh:
        for r in existing:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in new_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[PASS] wrote {OUT_TREND30.name}: {len(trend_existing) + len(new_rows)} trend rows")
    print(f"[PASS] wrote {OUT_GOLD130.name}: {len(existing) + len(new_rows)} rows total")

    # Sanity check: every expected bucket has at least one event
    empty = [r for r in new_rows if not r["expected_aggregation"]["buckets"]]
    if empty:
        print(f"[WARN] {len(empty)} queries returned empty buckets; check filters")
    else:
        print("[PASS] every new query has non-empty expected buckets")


if __name__ == "__main__":
    main()

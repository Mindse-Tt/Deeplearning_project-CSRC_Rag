"""L6 Trend Aggregator — 结构化统计聚合层。

设计文档：prompts/trend_analyzer/aggregation_specs.yaml

当 Planner 识别为 ``trend_analysis`` 意图时，系统不走向量检索，而是：

1. 解析 query 中的维度（year / violation_type / punishment_type / agency）和窗口（year range）
2. 按维度在 ``event_corpus.jsonl`` 上做 groupby count
3. 把聚合结果渲染成 evidence_block（[Stat=...] 行格式）送给 Responder
4. 返回一个 ``TrendResult``，里面带原始数字 + 典型样例 event_id

这样 Responder 拿到的不是一堆 case snippets，而是"2022 年 387 起 / 2023 年 412 起"这样的硬数字，能给出真正的趋势分析答案。

这是我们对"统计类问题不该走向量检索"这一判断的核心落地：用 SQL-like 的 groupby
聚合替代语义召回，从根上保证趋势答案的数字准确性与可追溯性（每个统计项都带样例 EventID）。
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FacetValue:
    """One row of an aggregation result."""

    key: str
    count: int
    share: float  # 0..1
    sample_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AggregationSlice:
    """All rows for one facet (year / violation_type / ...)."""

    facet: str
    values: tuple[FacetValue, ...]
    total: int
    description: str = ""


@dataclass(frozen=True)
class TrendResult:
    """Complete aggregation output for a trend_analysis query."""

    query: str
    detected_facets: tuple[str, ...]
    year_window: tuple[int, int] | None
    slices: tuple[AggregationSlice, ...]
    evidence_block: str  # rendered text for Responder
    supporting_event_ids: tuple[str, ...]  # dedup'd across slices, for L7 validator


# ---------------------------------------------------------------------------
# Query-level facet detection
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")
_YEAR_RANGE_RE = re.compile(
    r"(19\d{2}|20\d{2})\s*[-–至到]\s*(19\d{2}|20\d{2})"
)
_RECENT_N_RE = re.compile(r"近\s*(\d{1,2})\s*年")

_FACET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "year": ("年度", "年份", "历年", "每年", "各年", "逐年", "年变化", "年走势"),
    "violation_type": (
        "违规类型",
        "违规种类",
        "违法类型",
        "违法种类",
        "违规分布",
        "什么类型",
        "哪类",
        "哪种类型",
        "类型分布",
    ),
    "punishment_type": (
        "处罚方式",
        "处罚类型",
        "处罚分布",
        "罚款",
        "警告",
        "市场禁入",
        "没收",
        "处分措施",
    ),
    "agency": ("机构", "证监局", "证监会", "交易所", "协会", "谁处罚", "哪个机构"),
}


def detect_facets(query: str) -> tuple[str, ...]:
    """Guess which facets the query is asking about.

    Returns a tuple preserving a stable order: year / violation_type /
    punishment_type / agency. Always includes ``year`` if the query
    mentions any year or "近 N 年" phrase.
    """
    # 维度识别：按关键词词典命中确定 groupby 维度（年度/违规类型/处罚方式/机构）。
    facets: list[str] = []
    for facet, keywords in _FACET_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            facets.append(facet)
    # 隐式年度：query 里出现具体年份或"近 N 年"时，即便没说"年度"也补上 year 维度。
    # Implicit year when query mentions a year literal
    if "year" not in facets and (_YEAR_RE.search(query) or _RECENT_N_RE.search(query)):
        facets.insert(0, "year")
    # Default facet: if the query is a "趋势" question with no explicit
    # facet, assume by year.
    if not facets and any(kw in query for kw in ("趋势", "变化", "走势", "统计")):
        facets = ["year"]
    return tuple(facets)


def detect_year_window(query: str, *, fallback: tuple[int, int] = (2016, 2025)) -> tuple[int, int] | None:
    """Extract ``(start, end)`` years from the query.

    Priority: explicit range → single year → "近 N 年" → None (caller's
    fallback applies).
    """
    m = _YEAR_RANGE_RE.search(query)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m = _RECENT_N_RE.search(query)
    if m:
        n = max(1, min(int(m.group(1)), 30))
        import datetime

        end = datetime.date.today().year
        return (end - n + 1, end)
    years = [int(y) for y in _YEAR_RE.findall(query)]
    if years:
        y0 = min(years)
        y1 = max(years)
        # Single year → one-year window
        return (y0, y1)
    # No explicit year at all → ``None`` so caller can decide
    return None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _year_of(event: dict[str, Any]) -> str | None:
    """Return a 4-char year string. Prefer declare_date over supervision_date."""
    for key in ("declare_date", "supervision_date"):
        v = event.get(key)
        if v and isinstance(v, str) and len(v) >= 4:
            return v[:4]
    return None


_AGENCY_RE = re.compile(
    r"(中国证监会|[^\s（(]*?证监局|上海证券交易所|深圳证券交易所|北京证券交易所|上交所|深交所|北交所|证券业协会|基金业协会)"
)


def _agency_of(event: dict[str, Any]) -> str | None:
    """Normalise ``supervisor`` to a short agency label."""
    text = str(event.get("supervisor") or event.get("promulgator") or "")
    m = _AGENCY_RE.search(text)
    if m:
        return m.group(1)
    return text.split("(")[0].strip() or None


def _violation_types_of(event: dict[str, Any]) -> list[str]:
    raw = event.get("violation_types") or []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    flat: list[str] = []
    for item in raw:
        if not item:
            continue
        for sub in re.split(r"[;、；,，]", str(item)):
            sub = sub.strip()
            if sub:
                flat.append(sub)
    return flat


def _punishment_types_of(event: dict[str, Any]) -> list[str]:
    raw = event.get("punishment_types") or []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    return [str(x) for x in raw if x]


def _in_window(event: dict[str, Any], window: tuple[int, int] | None) -> bool:
    if window is None:
        return True
    year_str = _year_of(event)
    if not year_str:
        return False
    try:
        y = int(year_str)
    except ValueError:
        return False
    return window[0] <= y <= window[1]


def _pick_samples(
    event_ids: Sequence[str], k: int = 2
) -> tuple[str, ...]:
    """Pick at most k representative event_ids, preserving insertion order."""
    dedup: list[str] = []
    seen: set[str] = set()
    for eid in event_ids:
        if eid and eid not in seen:
            seen.add(eid)
            dedup.append(eid)
        if len(dedup) >= k:
            break
    return tuple(dedup)


def _aggregate_by(
    events: list[dict[str, Any]],
    facet: str,
    *,
    top_n: int | None = None,
) -> AggregationSlice:
    """Group events by a single facet and return an ordered slice.

    For multi-label facets (``violation_type`` / ``punishment_type``)
    share is computed against the **sum of labels**, not the number of
    distinct events — otherwise shares can exceed 100% when an event
    carries multiple labels. Year/agency are single-label so both
    denominators agree.

    核心：这是 SQL groupby count 的等价实现——按 facet 把事件分桶、统计每桶数量与占比，
    并保留每桶前若干个样例 EventID 用于 L7 引用校验。多标签维度（违规类型/处罚方式）
    一个事件可计入多个桶，故占比分母用"标签总数"而非"事件数"，避免占比超过 100%。
    """
    bucket_events: dict[str, list[str]] = defaultdict(list)
    event_coverage = 0
    label_total = 0
    for event in events:
        eid = str(event.get("event_id") or "")
        if facet == "year":
            key = _year_of(event)
            keys = [key] if key else []
        elif facet == "agency":
            key = _agency_of(event)
            keys = [key] if key else []
        elif facet == "violation_type":
            keys = _violation_types_of(event)
        elif facet == "punishment_type":
            keys = _punishment_types_of(event)
        else:
            keys = []
        if not keys:
            continue
        event_coverage += 1
        for k in keys:
            bucket_events[k].append(eid)
            label_total += 1

    # 排序策略：年度按时间正序（呈现趋势曲线），其余维度按数量降序（突出主要类别）。
    if facet == "year":
        ordered_keys = sorted(bucket_events.keys())
    else:
        ordered_keys = sorted(
            bucket_events.keys(),
            key=lambda k: (-len(bucket_events[k]), k),
        )

    if top_n is not None and facet != "year":
        ordered_keys = ordered_keys[:top_n]

    # Use label_total as denominator for multi-label facets so shares
    # can be interpreted as "X% of all labels attached to events".
    denom = label_total if facet in {"violation_type", "punishment_type"} else event_coverage

    values = []
    for k in ordered_keys:
        hits = bucket_events[k]
        count = len(hits)
        share = count / denom if denom else 0.0
        values.append(
            FacetValue(
                key=k,
                count=count,
                share=round(share, 4),
                sample_event_ids=_pick_samples(hits),
            )
        )

    return AggregationSlice(
        facet=facet,
        values=tuple(values),
        total=event_coverage,
        description=_FACET_DESCRIPTIONS.get(facet, facet),
    )


_FACET_DESCRIPTIONS: dict[str, str] = {
    "year": "年度处罚数量分布",
    "violation_type": "违规类型分布",
    "punishment_type": "处罚方式分布",
    "agency": "处罚机构排名",
}


# ---------------------------------------------------------------------------
# Evidence block rendering
# ---------------------------------------------------------------------------


def _render_evidence_block(
    slices: Sequence[AggregationSlice],
    *,
    year_window: tuple[int, int] | None,
    total_events: int,
    slot_filters: dict[str, list[str]] | None = None,
) -> str:
    lines: list[str] = []
    window_tag = (
        f"（窗口：{year_window[0]}-{year_window[1]} 年）" if year_window else "（全量）"
    )
    filter_tag = ""
    if slot_filters:
        constraints = []
        for key, values in slot_filters.items():
            if values:
                constraints.append(f"{key}={'/'.join(str(v) for v in values)}")
        if constraints:
            filter_tag = f"（约束：{'；'.join(constraints)}）"
    lines.append(f"[Stat=总量] 匹配事件 {total_events} 起{window_tag}{filter_tag}")
    for sl in slices:
        lines.append("")
        lines.append(f"[Facet={sl.facet}] {sl.description} (基数={sl.total})")
        for v in sl.values:
            share_pct = round(v.share * 100, 1)
            samples = "、".join(v.sample_event_ids) if v.sample_event_ids else "-"
            lines.append(
                f"  {sl.facet}={v.key}  count={v.count}  share={share_pct}%  样例EventID={samples}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TrendAggregator:
    """Stateless aggregator over the event corpus.

    Load once at engine startup; call :meth:`aggregate` per query.
    """

    def __init__(self, event_corpus: list[dict[str, Any]]) -> None:
        self._corpus = event_corpus

    @classmethod
    def from_jsonl(cls, path: Path) -> "TrendAggregator":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return cls(rows)

    def aggregate(
        self,
        query: str,
        *,
        facets: Sequence[str] | None = None,
        year_window: tuple[int, int] | None = None,
        top_n: int = 8,
        slot_filters: dict[str, list[str]] | None = None,
    ) -> TrendResult:
        """Run a structured aggregation for ``query`` and return a
        ``TrendResult`` suitable for the Responder evidence block.

        ``slot_filters`` is a dict of pre-extracted slot constraints
        that should further filter the event pool before aggregation.
        Supported keys: ``violation_type`` / ``punishment_type`` (values
        are lists; an event passes if it carries any of them). Typically
        the engine fills this from ``slot_filler.extract_slots(query)``.
        """
        # 聚合主流程：定维度 → 定年窗 → 先按年窗与槽位约束过滤事件池 → 逐维度 groupby。
        facets = tuple(facets) if facets else detect_facets(query)
        if not facets:
            facets = ("year",)
        if year_window is None:
            year_window = detect_year_window(query)

        # 第一道过滤：仅保留落在年窗内的事件（None 年窗表示全量）。
        filtered = [e for e in self._corpus if _in_window(e, year_window)]

        # Slot-based filtering (in addition to year window). Only applied
        # when explicit constraints are passed; detect_facets doesn't
        # trigger this because we want the user's violation_type slot to
        # _narrow_ the pool rather than _group_ it.
        if slot_filters:
            vt_values = slot_filters.get("violation_type") or []
            pt_values = slot_filters.get("punishment_type") or []
            if vt_values:
                needles = [str(v) for v in vt_values]
                filtered = [
                    e
                    for e in filtered
                    if any(
                        any(n in vt for vt in _violation_types_of(e))
                        for n in needles
                    )
                ]
            if pt_values:
                needles = [str(v) for v in pt_values]
                filtered = [
                    e
                    for e in filtered
                    if any(
                        any(n in pt for pt in _punishment_types_of(e))
                        for n in needles
                    )
                ]

        slices: list[AggregationSlice] = []
        for facet in facets:
            sl = _aggregate_by(filtered, facet, top_n=top_n)
            slices.append(sl)

        evidence_block = _render_evidence_block(
            slices,
            year_window=year_window,
            total_events=len(filtered),
            slot_filters=slot_filters,
        )
        # Dedup supporting event_ids across all slices for the L7 validator.
        dedup_ids: list[str] = []
        seen: set[str] = set()
        for sl in slices:
            for v in sl.values:
                for eid in v.sample_event_ids:
                    if eid and eid not in seen:
                        seen.add(eid)
                        dedup_ids.append(eid)
        return TrendResult(
            query=query,
            detected_facets=facets,
            year_window=year_window,
            slices=tuple(slices),
            evidence_block=evidence_block,
            supporting_event_ids=tuple(dedup_ids),
        )


__all__ = [
    "FacetValue",
    "AggregationSlice",
    "TrendResult",
    "TrendAggregator",
    "detect_facets",
    "detect_year_window",
]

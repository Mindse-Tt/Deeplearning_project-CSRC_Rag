"""L3a 元数据过滤：把 L2 抽取的结构化槽位转成候选文档白名单（软/硬双模）。

本模块消费 L2 slot_filler 产出的结构化槽位，计算出通过过滤的 chunk_id 集合，
作为 ``allowed_doc_ids`` 下推给 BM25 / 稠密编码器，在打分前就缩小候选池。

我们的核心设计是"软硬结合、保召回优先"：
- ``year`` / ``violation_type`` / ``org``  → 硬过滤（精确相等 / 包含匹配）；
- ``company``                              → 软过滤（仅作为后续重排的 boost 提示）；
- 一旦硬过滤后白名单过小（< ``min_allowed_fallback``），立即降级为软过滤
  （返回 ``None`` ⇒ 放行全语料），宁可少过滤也不让召回坍塌为空。
  这条降级规则是 R2 阶段补上的"软"行为，修复了 M2b 过滤过狠导致召回归零的问题。

除预建的元数据索引外本模块无状态；engine 应在启动时只实例化一次 ``MetadataFilter``。

Interface
---------
    mf = MetadataFilter.from_chunks(chunks)
    decision = mf.apply(
        slots={"year": "2024", "violation_type": "信息披露违规", "org": "证监会"},
        slot_confidence={"year": 0.95, "violation_type": 0.80, "org": 0.90},
    )
    # decision.allowed_doc_ids  -> set[str] | None
    # decision.applied_filters  -> dict[str, str]
    # decision.fallback          -> bool  (True if degraded to soft)
    # decision.boost_hints       -> dict[str, str]  (e.g. {"company": "中金公司"})
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

LOGGER = logging.getLogger(__name__)


# 硬过滤后候选数低于此阈值即降级为软过滤，避免召回被收成空集。
DEFAULT_MIN_ALLOWED_FALLBACK = 20

# 槽位置信度阈值：低于此值的槽位不参与硬过滤，仅作软提示。
# 取 0.6 是在"精确收窄候选"与"保护召回"之间的折中默认值，可由配置覆盖。
DEFAULT_SLOT_CONFIDENCE_THRESHOLD = 0.6

# 槽位 → chunk 字段的过滤分工：前三者可硬过滤，company 只做软提示。
HARD_FILTER_SLOTS = ("year", "violation_type", "org")
SOFT_FILTER_SLOTS = ("company",)


@dataclass(frozen=True)
class FilterDecision:
    allowed_doc_ids: set[str] | None
    applied_filters: dict[str, str]
    boost_hints: dict[str, str]
    fallback: bool
    diagnostics: dict[str, int]


@dataclass
class ChunkMetaRow:
    chunk_id: str
    year: str | None
    promulgator: str
    supervisor: str
    violation_types: tuple[str, ...]
    title: str


class MetadataFilter:
    """Apply structured slot filters on the chunk corpus."""

    def __init__(
        self,
        rows: list[ChunkMetaRow],
        *,
        min_allowed_fallback: int = DEFAULT_MIN_ALLOWED_FALLBACK,
        slot_confidence_threshold: float = DEFAULT_SLOT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._rows = rows
        self._min_allowed_fallback = min_allowed_fallback
        self._slot_confidence_threshold = slot_confidence_threshold

    # ------------------------------------------------------------------ ctor
    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[Mapping[str, Any]],
        *,
        min_allowed_fallback: int = DEFAULT_MIN_ALLOWED_FALLBACK,
        slot_confidence_threshold: float = DEFAULT_SLOT_CONFIDENCE_THRESHOLD,
    ) -> "MetadataFilter":
        rows: list[ChunkMetaRow] = []
        for chunk in chunks:
            rows.append(
                ChunkMetaRow(
                    chunk_id=chunk["chunk_id"],
                    year=chunk.get("year"),
                    promulgator=chunk.get("promulgator") or "",
                    supervisor=chunk.get("supervisor") or "",
                    violation_types=tuple(chunk.get("violation_types") or []),
                    title=chunk.get("title") or "",
                )
            )
        return cls(
            rows,
            min_allowed_fallback=min_allowed_fallback,
            slot_confidence_threshold=slot_confidence_threshold,
        )

    # --------------------------------------------------------------- public
    def apply(
        self,
        slots: Mapping[str, Any] | None,
        *,
        slot_confidence: Mapping[str, float] | None = None,
    ) -> FilterDecision:
        """Apply hard/soft filters based on the slot dict.

        Parameters
        ----------
        slots: normalised slot values, e.g. {"year": "2024", "org": "证监会"}.
        slot_confidence: optional confidence per slot (0.0-1.0). Slots whose
            confidence is below ``slot_confidence_threshold`` are ignored for
            hard filtering but still used as soft boost hints.
        """
        slots = dict(slots or {})
        confidences = dict(slot_confidence or {})

        applied: dict[str, str] = {}
        boost_hints: dict[str, str] = {}
        trusted_hard_slots: dict[str, str] = {}

        for slot_name in HARD_FILTER_SLOTS:
            value = _normalise_slot_value(slots.get(slot_name))
            if value is None:
                continue
            # 缺省置信度按 1.0 处理（调用方未传 ⇒ 视为可信硬过滤）。
            conf = float(confidences.get(slot_name, 1.0))
            if conf < self._slot_confidence_threshold:
                # 低置信槽位：不参与硬过滤，降格为软提示，避免错误抽取误杀候选。
                boost_hints[slot_name] = value
                continue
            trusted_hard_slots[slot_name] = value
            applied[slot_name] = value

        for slot_name in SOFT_FILTER_SLOTS:
            value = _normalise_slot_value(slots.get(slot_name))
            if value is None:
                continue
            boost_hints[slot_name] = value

        # No hard filters -> no restriction.
        if not trusted_hard_slots:
            return FilterDecision(
                allowed_doc_ids=None,
                applied_filters={},
                boost_hints=boost_hints,
                fallback=False,
                diagnostics={"allowed": len(self._rows), "total": len(self._rows)},
            )

        allowed = {
            row.chunk_id
            for row in self._rows
            if self._row_matches(row, trusted_hard_slots)
        }

        diagnostics = {
            "allowed": len(allowed),
            "total": len(self._rows),
        }

        # 召回保护：硬过滤结果过小则整体降级，宁可不过滤也不让候选坍塌。
        if len(allowed) < self._min_allowed_fallback:
            LOGGER.info(
                "Metadata hard-filter returned %d < %d chunks; degrading to soft.",
                len(allowed),
                self._min_allowed_fallback,
            )
            # 把原本的硬过滤槽位转存为软提示，让下游重排仍能利用这些信号。
            for slot_name, value in trusted_hard_slots.items():
                boost_hints.setdefault(slot_name, value)
            return FilterDecision(
                allowed_doc_ids=None,
                applied_filters={},
                boost_hints=boost_hints,
                fallback=True,
                diagnostics=diagnostics,
            )

        return FilterDecision(
            allowed_doc_ids=allowed,
            applied_filters=applied,
            boost_hints=boost_hints,
            fallback=False,
            diagnostics=diagnostics,
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _row_matches(row: ChunkMetaRow, slots: Mapping[str, str]) -> bool:
        # 逐槽位匹配，全部满足才放行：year 用精确相等，violation_type 用包含匹配
        # （一个 chunk 可挂多个违规类型），org 同时在发布/处罚机构两字段上做包含。
        if "year" in slots and row.year != slots["year"]:
            return False
        if "violation_type" in slots:
            needle = slots["violation_type"]
            if not any(needle in vt for vt in row.violation_types):
                return False
        if "org" in slots:
            needle = slots["org"]
            if needle not in row.promulgator and needle not in row.supervisor:
                return False
        return True


def _normalise_slot_value(value: Any) -> str | None:
    """Strip / reject empty slot values. Returns None if unusable."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        # Slot filler may return a list; take the first non-empty entry.
        for item in value:
            norm = _normalise_slot_value(item)
            if norm is not None:
                return norm
        return None
    text = str(value).strip()
    if not text:
        return None
    return text

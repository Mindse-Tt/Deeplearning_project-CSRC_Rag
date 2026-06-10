"""L3 检索计划构建：把用户查询 + 意图编译成可执行的检索计划（QueryPlan）。

本模块位于意图识别之后、实际打分之前，负责把"自然语言查询"翻译成结构化的检索
参数：检索单元（chunk / event）、返回条数 top_k，以及从查询里轻量抽取的元数据
过滤项。这里只做基于正则/关键词的**确定性、低成本**抽取，作为 slot_filler 之外
的兜底信号——即使 slot_filler 漏抽，"2023 年""上市公司""证监会"这类高频显式线索
也能在此被捕获并下推为候选过滤条件。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from csrc_rag.orchestration.intents import IntentSpec


@dataclass(frozen=True)
class QueryPlan:
    # 检索计划的不可变快照，随整条 SearchResponse 回传，便于线上可观测与复现。
    intent: str
    retrieval_unit: str
    top_k: int
    metadata_filters: dict[str, str]
    query_text: str


# 年份线索：限定 19xx/20xx 四位年，避免把无关数字（如金额、案号）误判成年份。
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def build_query_plan(query: str, intent: IntentSpec) -> QueryPlan:
    # 轻量规则抽取元数据过滤项。这是与 slot_filler 互补的"显式线索"兜底通道，
    # 命中即下推为候选过滤；最终是否硬过滤由 engine 结合置信度统一决策。
    filters: dict[str, str] = {}
    year_match = YEAR_PATTERN.search(query)
    if year_match:
        filters["year"] = year_match.group(0)
    if "上市公司" in query:
        filters["is_listed_company"] = "1"
    if "证监会" in query:
        # 仅作 regulator_hint 提示而非强约束——"证监会"几乎命中全语料，
        # 强过滤会把候选池收得过窄（详见 engine 中的软过滤说明）。
        filters["regulator_hint"] = "证监会"
    return QueryPlan(
        intent=intent.name,
        retrieval_unit=intent.retrieval_unit,
        top_k=intent.top_k,
        metadata_filters=filters,
        query_text=query,
    )


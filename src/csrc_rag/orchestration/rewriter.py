"""L2 查询改写层 —— 共指消解 + 同义词扩展 + 槽位抽取的三段式流水线。

本模块承接 L1 意图分类的结果，把"依赖上下文的口语化 query"改写为
"可独立检索的规范化 query"，是多轮对话能稳定命中的关键。整体非侵入设计：
不强依赖 L3/L5，任何子步骤失败都安全降级回原始 query。

流水线（rewrite 为唯一主入口）：
    raw_query + 多轮 history
      → L2a 共指消解：先规则（代词→会话实体回填，带置信度打分），
        置信度不足时再走注入式 LLM 兜底改写；
      → L2b 同义词扩展：基于词典把 alias 归一到 canonical，产出扩展 query 供召回增强；
      → L2c 槽位抽取：复用 slot_filler，抽取年份/股票代码/违规类型等结构化约束；
      → 汇总为 RewriteOutput（含 canonical_query、扩展集、slots、过滤器、可追溯 trace）。

设计要点：greeting/chitchat/out_of_scope 等非 RAG 意图直接旁路，不做任何改写。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

from .slot_filler import extract_slots, slots_to_filters

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
_SYNONYMS_PATH = _CONFIG_ROOT / "synonyms.json"

HISTORY_WINDOW = 6          # 最多回看 6 轮
MAX_EXPANSIONS = 5          # 同义词扩展 query 最多 5 条
COREF_CONF_THRESHOLD = 0.8  # 共指置信度阈值

# 代词类别（正则 + 触发类型）
_PRONOUN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(那|这)(个|件|起|位)案(子|件)?"), "case"),
    (re.compile(r"(该|此)案(子|件)?"), "case"),
    (re.compile(r"(那|这)(家|个)公司"), "company"),
    (re.compile(r"(该|此)公司"), "company"),
    (re.compile(r"(那|这)(个|位)人"), "person"),
    (re.compile(r"该当事人"), "person"),
    (re.compile(r"(它|他|她|它们|他们|她们)"), "any"),
    (re.compile(r"^那它?的"), "any"),  # "那它的法条呢"
]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
@dataclass
class RewriteInput:
    raw_query: str
    history: list[dict[str, str]] = field(default_factory=list)  # [{role, content}]
    intent: Optional[str] = None
    session_entities: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewriteTrace:
    coref_triggered: bool = False
    coref_method: str = "none"          # rule | llm | none
    synonyms_hit: list[tuple[str, str]] = field(default_factory=list)
    slot_source: dict[str, str] = field(default_factory=dict)


@dataclass
class RewriteOutput:
    canonical_query: str
    synonyms_expanded: list[str] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, list] = field(default_factory=lambda: {"must": [], "should": []})
    rewrite_trace: RewriteTrace = field(default_factory=RewriteTrace)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Synonym dictionary
# ---------------------------------------------------------------------------
class SynonymDict:
    """Canonical -> list[aliases]; reverse index used at match time."""

    _cache: Optional["SynonymDict"] = None

    def __init__(self, mapping: dict[str, list[str]]):
        self.canonical_to_aliases = mapping
        # alias -> canonical (reverse index, longest-first)
        self.alias_to_canonical: dict[str, str] = {}
        for canon, aliases in mapping.items():
            for a in aliases:
                self.alias_to_canonical[a] = canon
        # sort aliases by length desc for longest-match
        self._alias_sorted = sorted(self.alias_to_canonical.keys(), key=len, reverse=True)

    @classmethod
    def load(cls, path: Path = _SYNONYMS_PATH) -> "SynonymDict":
        """Load synonyms.json supporting two schemas:

        - Flat: {"synonyms": {canonical: [aliases...]}}
        - Categorized (current repo): {"violation": {canon: [aliases]}, "sanction": {...}, ...}
        """
        if cls._cache is not None:
            return cls._cache
        mapping: dict[str, list[str]] = {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Strategy 1: flat "synonyms" key
                if isinstance(raw.get("synonyms"), dict):
                    mapping.update(raw["synonyms"])
                # Strategy 2: categorized — iterate top-level dicts, skip meta
                for key, val in raw.items():
                    if key.startswith("_") or key == "synonyms":
                        continue
                    if isinstance(val, dict):
                        for canon, aliases in val.items():
                            if isinstance(aliases, list):
                                mapping.setdefault(canon, []).extend(aliases)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load synonyms.json: %s", exc)
        cls._cache = cls(mapping)
        return cls._cache

    def expand(self, query: str, limit: int = MAX_EXPANSIONS) -> tuple[list[str], list[tuple[str, str]]]:
        """同义词扩展：把 query 中的别名替换为规范词，返回 (扩展集, 命中记录)。

        expanded_queries: 原 query 的 alias -> canonical 替换版本（每命中一个 alias 产生一条扩展）。
        hits: [(alias, canonical), ...] 用于 trace。
        """
        hits: list[tuple[str, str]] = []
        expansions: list[str] = []
        # 别名按长度降序遍历（最长匹配优先），避免短别名先吃掉长别名的子串。
        for alias in self._alias_sorted:
            if alias in query and alias != self.alias_to_canonical[alias]:
                canon = self.alias_to_canonical[alias]
                hits.append((alias, canon))
                rewritten = query.replace(alias, canon)
                if rewritten != query and rewritten not in expansions:
                    expansions.append(rewritten)
                if len(expansions) >= limit:
                    break
        return expansions, hits


# ---------------------------------------------------------------------------
# L2a Coreference Resolution
# ---------------------------------------------------------------------------
def _detect_pronoun(query: str) -> Optional[str]:
    for pat, ptype in _PRONOUN_PATTERNS:
        if pat.search(query):
            return ptype
    return None


def _recent_entities_from_history(history: list[dict[str, str]]) -> dict[str, Any]:
    """从 history 的最近若干轮回抽 entity（简化版：只看 slot_filler 能抽的）。

    真实版本应读取上一轮的 RewriteOutput.slots；这里对 assistant 回复和 user query 都跑一遍 slot_filler。
    """
    entities: dict[str, list] = {}
    window = history[-HISTORY_WINDOW:]
    for turn in reversed(window):  # 最近优先
        text = turn.get("content", "")
        slots, _src = extract_slots(text)
        for k, v in slots.items():
            if v and k not in entities:
                entities[k] = v
    return entities


def _rule_based_coref(
    raw_query: str,
    history: list[dict[str, str]],
    session_entities: dict[str, Any],
) -> tuple[Optional[str], float]:
    """返回 (rewritten_query, confidence)。

    confidence ∈ [0, 1]，≥ 0.8 才采纳。
    """
    # 无代词则无需消解，直接返回（confidence=0 表示不采纳）。
    ptype = _detect_pronoun(raw_query)
    if ptype is None:
        return None, 0.0

    # 实体来源：本轮显式 session_entities 优先，其次从最近 history 回抽，互补不覆盖。
    entities = dict(session_entities or {})
    for k, v in _recent_entities_from_history(history).items():
        entities.setdefault(k, v)

    if not entities:
        return None, 0.0

    # 按代词类型选择回填实体：company/person 精确匹配；case 优先公司其次人；
    # "any" 类按 company → person → stock_code 的强度顺序选第一个可用实体。
    # 选择回填实体
    target_field: Optional[str] = None
    if ptype == "company" and entities.get("company"):
        target_field = "company"
    elif ptype == "person" and entities.get("person"):
        target_field = "person"
    elif ptype == "case":
        # 案件层面：优先 company，其次 person
        target_field = "company" if entities.get("company") else ("person" if entities.get("person") else None)
    else:  # "any"
        for candidate in ("company", "person", "stock_code"):
            if entities.get(candidate):
                target_field = candidate
                break

    if not target_field:
        return None, 0.0

    entity_val = entities[target_field]
    entity_str = entity_val[0] if isinstance(entity_val, list) else str(entity_val)

    # 置信度打分：类型匹配给基础分，再叠加"对话短/强实体/槽位唯一"等增益信号；
    # 仅当总分 ≥ COREF_CONF_THRESHOLD(0.8) 时才采纳规则结果，否则交给 LLM 兜底。
    conf = 0.3  # 基础分（类型匹配）
    if len(history) <= 3:
        conf += 0.2
    if target_field in ("company", "stock_code"):  # 更强的实体
        conf += 0.3
    if isinstance(entity_val, list) and len(entity_val) == 1:
        conf += 0.2  # 唯一槽位

    # 替换代词为实体名（粗糙版：把第一个匹配的代词短语替换掉；"那它的法条呢" -> "<entity> 的法条")
    rewritten = raw_query
    for pat, _ in _PRONOUN_PATTERNS:
        if pat.search(rewritten):
            rewritten = pat.sub(entity_str, rewritten, count=1)
            break

    # 同时把 session 的其他强实体附加到 query 末尾（让 BM25 更稳）
    extras = []
    for k in ("year", "violation_type"):
        v = entities.get(k)
        if v:
            extras.append(str(v[0]) if isinstance(v, list) else str(v))
    if extras:
        rewritten = f"{rewritten} {' '.join(extras)}"

    return rewritten, min(conf, 1.0)


def _llm_fallback_coref(
    raw_query: str,
    history: list[dict[str, str]],
    llm_fn: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """LLM 兜底改写。llm_fn 由调用方注入（解耦 Qwen 依赖），超时/异常返回 None。"""
    if llm_fn is None:
        return None
    recent = history[-3:] if history else []
    ctx = "\n".join(f"{t.get('role','user')}: {t.get('content','')}" for t in recent)
    prompt = (
        "你是 query 改写助手。根据最近对话，把当前问题改写为一个不依赖上下文、可独立检索的完整问题。\n"
        "要求：保留原意、不要编造、不要回答问题。只输出改写后的问题。\n"
        f"{ctx}\n"
        f"当前问题：{raw_query}\n"
        "改写后："
    )
    try:
        out = llm_fn(prompt)
        if isinstance(out, str) and out.strip():
            return out.strip().splitlines()[0][:200]
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM coref fallback failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def rewrite(
    raw_query: str,
    history: Optional[list[dict[str, str]]] = None,
    intent: Optional[str] = None,
    session_entities: Optional[dict[str, Any]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> RewriteOutput:
    """L2 主入口。

    Parameters
    ----------
    raw_query : 用户当前轮原始输入
    history   : 最近若干轮对话 [{role, content}]
    intent    : L1 传下来的意图；若为 greeting/chitchat/out_of_scope 直接旁路
    session_entities : 上一轮固化的实体（可选）
    llm_fn    : 注入式 LLM 调用，签名 (prompt) -> str；缺省则跳过 LLM fallback
    """
    history = history or []
    session_entities = session_entities or {}
    trace = RewriteTrace()

    # 非 RAG 意图直接旁路
    if intent in {"greeting", "chitchat", "out_of_scope"}:
        return RewriteOutput(canonical_query=raw_query, rewrite_trace=trace)

    # ----- L2a 共指消解 -----
    # 三级采纳策略：规则高置信 → LLM 兜底 → 规则低置信仍优于原文。
    canonical = raw_query
    if _detect_pronoun(raw_query) is not None:
        trace.coref_triggered = True
        rule_out, conf = _rule_based_coref(raw_query, history, session_entities)
        if rule_out and conf >= COREF_CONF_THRESHOLD:
            canonical = rule_out
            trace.coref_method = "rule"
        else:
            llm_out = _llm_fallback_coref(raw_query, history, llm_fn)
            if llm_out:
                canonical = llm_out
                trace.coref_method = "llm"
            elif rule_out:  # 规则给出但置信度不够，也比纯 raw 好
                canonical = rule_out
                trace.coref_method = "rule"

    # ----- L2b 同义词扩展 -----
    syn_dict = SynonymDict.load()
    expansions, hits = syn_dict.expand(canonical, limit=MAX_EXPANSIONS)
    trace.synonyms_hit = hits

    # ----- L2c 槽位抽取 -----
    slots, slot_src = extract_slots(canonical)
    trace.slot_source = slot_src
    filters = slots_to_filters(slots)

    return RewriteOutput(
        canonical_query=canonical,
        synonyms_expanded=expansions,
        slots=slots,
        filters=filters,
        rewrite_trace=trace,
    )


__all__ = [
    "RewriteInput",
    "RewriteOutput",
    "RewriteTrace",
    "SynonymDict",
    "rewrite",
]

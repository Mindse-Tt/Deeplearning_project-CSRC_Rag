"""L2c 槽位填充层 —— 正则 + 词典 +（可选）NER 的结构化槽位抽取。

本模块从规范化 query 中抽取可用于硬过滤/加权的结构化约束，是把"自然语言问题"
转成"检索过滤条件"的桥梁。被 rewriter（L2）调用，输出再喂给 L3 检索与 L6 趋势聚合。

抽取字段：year / stock_code / violation_type / institution / company / person。
设计原则：抽不到即返回 None，绝不阻塞主流程（槽位是增强项而非必需项）。
各字段采用差异化策略——年份/股票代码走精确正则，违规类型/机构走词典匹配，
公司名当前为正则占位（预留 NER 接口），并通过 slot_source 标注每个槽位的来源便于追溯。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
_SYNONYMS_PATH = _CONFIG_ROOT / "synonyms.json"

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------
_RE_YEAR_ABS = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
_RE_YEAR_RECENT = re.compile(r"近\s*(\d{1,2})\s*年")
_RE_YEAR_AMOUNT_BLOCK = re.compile(r"(19\d{2}|20\d{2})\s*(元|万|亿|美元|港币)")  # 否决：金额别当年份
_RE_STOCK_CODE = re.compile(r"(?<!\d)(60\d{4}|000\d{3}|001\d{3}|002\d{3}|003\d{3}|300\d{3}|301\d{3}|688\d{3}|689\d{3})(?!\d)")
# 机构（精选白名单，按最长优先）
_INSTITUTION_KEYWORDS = [
    "中国证券监督管理委员会", "中国证监会", "证监会",
    "北京证监局", "上海证监局", "深圳证监局", "广东证监局", "浙江证监局",
    "江苏证监局", "山东证监局", "四川证监局", "湖北证监局", "福建证监局",
    "上海证券交易所", "深圳证券交易所", "北京证券交易所",
    "上交所", "深交所", "北交所",
]
# 人名粗筛（中文 2-4 字，后跟"某""先生""女士"等弱指示）
_RE_PERSON_CN = re.compile(r"([\u4e00-\u9fa5]{1,2}某(某)?)")


# ---------------------------------------------------------------------------
# Lazy-loaded violation type dictionary (reuse synonyms canonicals)
# ---------------------------------------------------------------------------
_VIOLATION_CACHE: Optional[list[str]] = None
_VIOLATION_ALIAS_CACHE: Optional[dict[str, str]] = None


def _load_violation_types() -> tuple[list[str], dict[str, str]]:
    global _VIOLATION_CACHE, _VIOLATION_ALIAS_CACHE
    if _VIOLATION_CACHE is not None and _VIOLATION_ALIAS_CACHE is not None:
        return _VIOLATION_CACHE, _VIOLATION_ALIAS_CACHE
    violations: list[str] = []
    alias_map: dict[str, str] = {}
    try:
        raw = json.loads(_SYNONYMS_PATH.read_text(encoding="utf-8"))
        # 优先 categorized: raw["violation"][canon] = [aliases]
        vio_block = raw.get("violation") if isinstance(raw, dict) else None
        if isinstance(vio_block, dict):
            for canon, aliases in vio_block.items():
                violations.append(canon)
                alias_map[canon] = canon
                if isinstance(aliases, list):
                    for a in aliases:
                        alias_map[a] = canon
        else:
            # Fallback: flat mapping + explicit violation_types list
            mapping = raw.get("synonyms", {}) if isinstance(raw, dict) else {}
            vio_set = set(raw.get("violation_types", [])) if isinstance(raw, dict) else set()
            for canon, aliases in mapping.items():
                if not vio_set or canon in vio_set:
                    violations.append(canon)
                    alias_map[canon] = canon
                    for a in aliases:
                        alias_map[a] = canon
    except Exception as exc:  # noqa: BLE001
        logger.warning("slot_filler: failed to load synonyms for violation dict: %s", exc)
    _VIOLATION_CACHE = sorted(set(violations), key=len, reverse=True)
    _VIOLATION_ALIAS_CACHE = dict(sorted(alias_map.items(), key=lambda kv: -len(kv[0])))
    return _VIOLATION_CACHE, _VIOLATION_ALIAS_CACHE


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------
def _extract_year(text: str) -> list[int]:
    # 先标记"金额后缀紧跟的四位数"区间，避免把"2000 万"里的 2000 误判成年份。
    blocked_spans = [m.span(1) for m in _RE_YEAR_AMOUNT_BLOCK.finditer(text)]
    years: list[int] = []
    for m in _RE_YEAR_ABS.finditer(text):
        if any(m.span() == b for b in blocked_spans):
            continue
        y = int(m.group(1))
        if 1990 <= y <= 2030 and y not in years:
            years.append(y)
    # "近 N 年" -> 展开为区间（以当前年为参考动态计算，避免硬编码当前年）
    import datetime
    this_year = datetime.date.today().year
    for m in _RE_YEAR_RECENT.finditer(text):
        n = int(m.group(1))
        n = max(1, min(n, 10))
        for y in range(this_year - n + 1, this_year + 1):
            if y not in years:
                years.append(y)
    return years


def _extract_stock_codes(text: str) -> list[str]:
    seen = []
    for m in _RE_STOCK_CODE.finditer(text):
        code = m.group(1)
        if code not in seen:
            seen.append(code)
    return seen


def _extract_violation_types(text: str) -> list[str]:
    _, alias_map = _load_violation_types()
    hits: list[str] = []
    for alias, canon in alias_map.items():
        if alias in text and canon not in hits:
            hits.append(canon)
    return hits


def _extract_institutions(text: str) -> list[str]:
    hits: list[str] = []
    for kw in sorted(_INSTITUTION_KEYWORDS, key=len, reverse=True):
        if kw in text and kw not in hits:
            hits.append(kw)
    return hits


def _extract_persons(text: str) -> list[str]:
    hits: list[str] = []
    for m in _RE_PERSON_CN.finditer(text):
        name = m.group(1)
        if name not in hits:
            hits.append(name)
    return hits


def _extract_companies_stub(text: str) -> list[str]:
    """公司名抽取占位：真实实现应接 LLM NER 或 jieba nr 词性。

    目前：匹配 "XX 公司 / XX 证券 / XX 股份" 等简单 pattern。
    """
    pat = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,10}(?:证券|公司|股份|集团|基金|银行|保险|信托))")
    hits: list[str] = []
    for m in pat.finditer(text):
        name = m.group(1)
        if len(name) >= 3 and name not in hits:
            hits.append(name)
    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_slots(text: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (slots, slot_source).

    slots: {year, stock_code, violation_type, institution, company, person}
    slot_source: {field: 'regex'|'dict'|'ner'|'none'}
    """
    slots: dict[str, Any] = {}
    src: dict[str, str] = {}

    if (ys := _extract_year(text)):
        slots["year"] = ys
        src["year"] = "regex"
    if (cs := _extract_stock_codes(text)):
        slots["stock_code"] = cs
        src["stock_code"] = "regex"
    if (vt := _extract_violation_types(text)):
        slots["violation_type"] = vt
        src["violation_type"] = "dict"
    if (ins := _extract_institutions(text)):
        slots["institution"] = ins
        src["institution"] = "dict"
    if (cos := _extract_companies_stub(text)):
        slots["company"] = cos
        src["company"] = "regex"  # stub；真实版 'ner'
    if (ps := _extract_persons(text)):
        slots["person"] = ps
        src["person"] = "regex"

    return slots, src


def slots_to_filters(slots: dict[str, Any]) -> dict[str, list[dict]]:
    """把抽取到的槽位转换为 L3a 检索过滤器（区分 must 强约束与 should 软加权）。

    分级理由：股票代码/年份歧义低、可靠，作 must 强过滤精确缩小候选集；
    违规类型/机构作 should 加权召回；公司/人名因别名变体多、易误伤，
    同样只作 should 软约束，避免把正确文档过滤掉。

    - stock_code / year → must（强过滤）
    - violation_type / institution → should（加权）
    - company / person → should（变体多，不强过滤）
    """
    must: list[dict] = []
    should: list[dict] = []

    if slots.get("year"):
        must.append({"field": "year", "op": "in", "value": list(slots["year"])})
    if slots.get("stock_code"):
        codes = list(slots["stock_code"])
        op = "eq" if len(codes) == 1 else "in"
        val: Any = codes[0] if len(codes) == 1 else codes
        must.append({"field": "stock_code", "op": op, "value": val})
    if slots.get("violation_type"):
        should.append({"field": "violation_type", "op": "in", "value": list(slots["violation_type"])})
    if slots.get("institution"):
        should.append({"field": "institution", "op": "in", "value": list(slots["institution"])})
    if slots.get("company"):
        should.append({"field": "company", "op": "in", "value": list(slots["company"])})
    if slots.get("person"):
        should.append({"field": "person", "op": "in", "value": list(slots["person"])})

    return {"must": must, "should": should}


__all__ = ["extract_slots", "slots_to_filters"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_gold_extra.py

Synthesize 60 **single-constraint, single-gold** evaluation queries from
``data/processed/event_corpus.jsonl`` and write them to
``data/eval/gold_extra_candidates.jsonl``.

Design (M3e gold_100 expansion):

* Intent split: ~80 %% ``case_retrieval`` + ~20 %% ``law_grounding``
  (only these two intents are emitted — no ``sanction_recommendation`` /
  ``trend_analysis`` to keep gold tight).
* Each row references exactly ONE ``event_id``. The event is picked so that
  its single distinctive keyword (unique person / unique company name) is
  literally present in ``event.activity``.
* Single-constraint query: one year OR one violation type OR one party
  OR one company (no AND composition).
* Balanced across the major violation-type buckets observed in the
  corpus (内幕交易 / 虚假记载 / 推迟披露 / 违规买卖股票 / 操纵股价 /
  占用公司资产 / 违规担保 / 重大遗漏 / 虚构利润).

Constraints enforced at build time (hard-fail the row otherwise):

1. ``relevant_event_ids`` length == 1 and the id exists in the corpus.
2. The query's anchor keyword (person / company name) is a substring of
   ``event.activity``.
3. Query length ∈ [10, 40] Chinese characters.
4. ``expected_slots`` has ≥ 1 non-empty value.
5. ``is_trap`` == False, ``difficulty`` ∈ {easy, medium}.

Stdlib-only (json / random / collections / re / pathlib).
"""
from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "processed" / "event_corpus.jsonl"
OUT_PATH = ROOT / "data" / "eval" / "gold_extra_candidates.jsonl"

SEED = 20260422
TARGET_TOTAL = 60
TARGET_CASE = 48  # case_retrieval
TARGET_LAW = 12   # law_grounding

# ------- Violation-type canonical buckets (fallback laws) -------
VTYPE_FALLBACK_LAWS: dict[str, list[str]] = {
    "内幕交易": ["《证券法》第五十条", "《证券法》第五十三条第一款"],
    "虚构利润": ["《证券法》第七十八条第二款", "2005年《证券法》第六十三条"],
    "虚假记载(误导性陈述)": ["《证券法》第七十八条第二款", "2005年《证券法》第六十三条"],
    "推迟披露": ["《证券法》第七十八条第一款", "《证券法》第八十条"],
    "违规买卖股票": ["《证券法》第四十四条", "2005年《证券法》第八十六条"],
    "操纵股价": ["《证券法》第五十五条第一款", "2005年《证券法》第七十七条"],
    "占用公司资产": ["《证券法》第七十八条", "《证券法》第八十条"],
    "违规担保": ["《证券法》第七十八条第二款", "2005年《证券法》第六十三条"],
    "重大遗漏": ["《证券法》第七十八条第一款", "2005年《证券法》第六十三条"],
    "一般会计处理不当": ["《证券法》第七十八条第二款"],
    "披露不实(其它)": ["《证券法》第七十八条第二款"],
    "欺诈上市": ["《证券法》第一百八十一条"],
    "其他": ["《证券法》第七十八条"],
}

VTYPE_PRIORITY = [
    "内幕交易",
    "操纵股价",
    "虚构利润",
    "虚假记载(误导性陈述)",
    "推迟披露",
    "违规买卖股票",
    "占用公司资产",
    "违规担保",
    "重大遗漏",
    "一般会计处理不当",
    "披露不实(其它)",
    "欺诈上市",
]

# Target count per bucket (case_retrieval pool; law_grounding reuses remaining).
CASE_TARGET_PER_BUCKET: dict[str, int] = {
    "内幕交易": 8,
    "虚假记载(误导性陈述)": 8,
    "推迟披露": 7,
    "违规买卖股票": 7,
    "虚构利润": 6,
    "操纵股价": 4,
    "占用公司资产": 3,
    "违规担保": 3,
    "重大遗漏": 2,
}


# ---------------------- helpers ----------------------


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def split_vtypes(vtypes: Iterable[str] | None) -> set[str]:
    out: set[str] = set()
    for v in (vtypes or []):
        for seg in re.split(r"[;；、,]", str(v)):
            seg = seg.strip()
            if seg:
                out.add(seg)
    return out


def primary_vtype(event: dict) -> str:
    vs = split_vtypes(event.get("violation_types"))
    for p in VTYPE_PRIORITY:
        if p in vs:
            return p
    return next(iter(vs)) if vs else "其他"


def short_vtype_label(vtype: str) -> str:
    """Shorten "虚假记载(误导性陈述)" -> "虚假记载" for compact queries."""
    return re.sub(r"\(.*?\)", "", vtype).strip() or vtype


def is_person_name(name: str) -> bool:
    if not name:
        return False
    if any(tok in name for tok in [
        "公司", "集团", "股份", "事务所", "研究院", "研究所",
        "合伙", "有限", "中心", "银行", "保险", "基金", "证券",
    ]):
        return False
    return 2 <= len(name) <= 5


def is_company_name(name: str) -> bool:
    if not name or len(name) < 4:
        return False
    return any(tok in name for tok in [
        "公司", "集团", "股份", "事务所", "研究院", "研究所", "合伙",
    ])


def year_of(event: dict) -> str:
    for key in ("supervision_date", "declare_date"):
        v = event.get(key) or ""
        if v and len(v) >= 4 and v[:4].isdigit():
            return v[:4]
    return ""


def first_position(event: dict) -> str:
    for ps in (event.get("positions") or []):
        for seg in str(ps).split(","):
            seg = seg.strip()
            if seg:
                return seg
    return ""


def derive_laws(event: dict, vtype: str) -> list[str]:
    """Pull 1-2 laws from event.law if recognizable, else fall back by vtype."""
    law_text = event.get("law") or ""
    found: list[str] = []
    # Regex: 《证券法》/2005年《证券法》 + 第X条 + 可选第X款 + 可选第X项
    pattern = re.compile(
        r"(?:2005年)?《中华人民共和国证券法》|(?:2005年)?《证券法》|《中国注册会计师审计准则第\d+号》"
    )
    section_re = re.compile(r"第[一二三四五六七八九十百零]+条(?:第[一二三四五六七八九十]+款)?(?:第[一二三四五六七八九十]+项)?")
    for m in pattern.finditer(law_text):
        start = m.end()
        tail = law_text[start:start + 30]
        sec = section_re.search(tail)
        if sec:
            base = m.group(0).replace("《中华人民共和国证券法》", "《证券法》")
            found.append(f"{base}{sec.group(0)}")
        if len(found) >= 2:
            break
    if found:
        # de-dup preserving order
        seen = set()
        uniq = []
        for f in found:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        return uniq[:2]
    return list(VTYPE_FALLBACK_LAWS.get(vtype, VTYPE_FALLBACK_LAWS["其他"]))[:2]


# ------------- query generators (single-constraint) -------------


def make_case_query(kind: str, name: str, event: dict, vtype: str) -> tuple[str, dict]:
    """Return (query_text, expected_slots). Single anchor = name (company/person)."""
    short_vt = short_vtype_label(vtype)
    year = year_of(event)
    slots: dict[str, str | int] = {}
    if kind == "person":
        pos = first_position(event)
        if pos:
            slots["party_role"] = pos
        slots["violation_type"] = short_vt
        # use role to make query a bit more informative when available
        if pos:
            q = f"{name}（{pos}）因{short_vt}被证监会处罚的案件情况如何？"
        else:
            q = f"{name}因{short_vt}被证监会行政处罚的案件详情？"
    else:  # company
        slots["violation_type"] = short_vt
        if year:
            slots["year"] = int(year) if year.isdigit() else year
        q = f"{name}因{short_vt}被证监会处罚的案件是什么？"
    return q, slots


def make_law_query(kind: str, name: str, event: dict, vtype: str) -> tuple[str, dict]:
    short_vt = short_vtype_label(vtype)
    slots: dict[str, str] = {"violation_type": short_vt, "focus": "law"}
    if kind == "person":
        pos = first_position(event)
        if pos:
            slots["party_role"] = pos
        q = f"{name}{short_vt}案援引了《证券法》哪一条？"
    else:
        q = f"{name}{short_vt}案援引了哪些法律条款？"
    return q, slots


def make_keypoints_case(name: str, vtype: str, event: dict) -> list[str]:
    short_vt = short_vtype_label(vtype)
    eid = event.get("event_id")
    kps = [
        f"必须给出 EventID {eid}",
        f"必须点出违规类型包含'{short_vt}'",
    ]
    pun = event.get("punishment_types") or []
    if pun:
        kps.append(f"必须说明处罚方式至少包含'{pun[0]}'")
    return kps[:3]


def make_keypoints_law(name: str, vtype: str, event: dict, laws: list[str]) -> list[str]:
    eid = event.get("event_id")
    first_law = laws[0] if laws else "《证券法》"
    kps = [
        f"必须给出 EventID {eid}",
        f"必须引用{first_law}",
    ]
    short_vt = short_vtype_label(vtype)
    kps.append(f"必须点出违规类型为'{short_vt}'")
    return kps[:3]


# ---------------------- main ----------------------


def main() -> int:
    rng = random.Random(SEED)
    events = load_events(CORPUS_PATH)

    # Index by name uniqueness
    person_index: dict[str, list[dict]] = collections.defaultdict(list)
    company_index: dict[str, list[dict]] = collections.defaultdict(list)
    for e in events:
        for p in (e.get("parties") or []):
            if is_person_name(p):
                person_index[p].append(e)
            if is_company_name(p):
                company_index[p].append(e)

    # Keep only names unique across corpus AND appearing literally in activity
    unique_persons: dict[str, dict] = {}
    for name, evs in person_index.items():
        if len(evs) != 1:
            continue
        e = evs[0]
        if name in (e.get("activity") or ""):
            unique_persons[name] = e

    unique_companies: dict[str, dict] = {}
    for name, evs in company_index.items():
        if len(evs) != 1:
            continue
        e = evs[0]
        act = e.get("activity") or ""
        # accept if full name OR a 4+-char distinctive prefix is in activity
        if name in act:
            unique_companies[name] = e
        else:
            # try prefix before "有限公司" / "股份有限公司"
            short = re.sub(r"(股份)?(有限)?(责任)?公司$", "", name)
            if len(short) >= 4 and short in act:
                unique_companies[name] = e

    print(f"[profile] unique_persons={len(unique_persons)}  unique_companies={len(unique_companies)}")

    # Bucketize by primary violation type
    buckets_person: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    buckets_company: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for name, e in unique_persons.items():
        buckets_person[primary_vtype(e)].append((name, e))
    for name, e in unique_companies.items():
        buckets_company[primary_vtype(e)].append((name, e))

    # Shuffle each bucket deterministically
    for b in (buckets_person, buckets_company):
        for k in b:
            b[k].sort(key=lambda t: t[1].get("event_id", ""))
            rng.shuffle(b[k])

    # Sample case_retrieval pool: aim for CASE_TARGET_PER_BUCKET; prefer person
    # when both kinds exist (person queries read more natural in Chinese).
    case_pool: list[tuple[str, str, dict, str]] = []  # (kind, name, event, vtype)
    used_event_ids: set[str] = set()
    for vtype, cap in CASE_TARGET_PER_BUCKET.items():
        picks: list[tuple[str, str, dict, str]] = []
        person_pool = buckets_person.get(vtype, [])
        company_pool = buckets_company.get(vtype, [])
        # Try person first, then fill with company
        i_p = 0
        i_c = 0
        while len(picks) < cap and (i_p < len(person_pool) or i_c < len(company_pool)):
            took = False
            if i_p < len(person_pool) and len(picks) < cap:
                n, e = person_pool[i_p]
                i_p += 1
                if e.get("event_id") not in used_event_ids:
                    picks.append(("person", n, e, vtype))
                    used_event_ids.add(e.get("event_id"))
                    took = True
            if i_c < len(company_pool) and len(picks) < cap:
                n, e = company_pool[i_c]
                i_c += 1
                if e.get("event_id") not in used_event_ids:
                    picks.append(("company", n, e, vtype))
                    used_event_ids.add(e.get("event_id"))
                    took = True
            if not took:
                break
        case_pool.extend(picks)

    # Top up to TARGET_CASE if we are short (some small buckets may underfill)
    if len(case_pool) < TARGET_CASE:
        for vtype in VTYPE_PRIORITY:
            if len(case_pool) >= TARGET_CASE:
                break
            for source in (buckets_person.get(vtype, []), buckets_company.get(vtype, [])):
                for n, e in source:
                    if len(case_pool) >= TARGET_CASE:
                        break
                    if e.get("event_id") in used_event_ids:
                        continue
                    kind = "person" if (source is buckets_person.get(vtype, [])) else "company"
                    case_pool.append((kind, n, e, vtype))
                    used_event_ids.add(e.get("event_id"))

    case_pool = case_pool[:TARGET_CASE]

    # Sample law_grounding pool: pick events where event.law has a recognizable
    # 《证券法》+第X条 citation (so the derived law is real, not fallback).
    # Diversify across vtypes by capping per-bucket count.
    LAW_CAP_PER_VTYPE = 2
    law_pool: list[tuple[str, str, dict, str]] = []
    law_vtype_count: collections.Counter[str] = collections.Counter()
    cite_re = re.compile(r"《证券法》第[一二三四五六七八九十]+条")

    def _try_add_law(kind: str, n: str, e: dict, vtype: str,
                     require_cite: bool) -> bool:
        if len(law_pool) >= TARGET_LAW:
            return False
        if law_vtype_count[vtype] >= LAW_CAP_PER_VTYPE:
            return False
        if e.get("event_id") in used_event_ids:
            return False
        if require_cite and not cite_re.search(e.get("law") or ""):
            return False
        law_pool.append((kind, n, e, vtype))
        law_vtype_count[vtype] += 1
        used_event_ids.add(e.get("event_id"))
        return True

    # Pass 1: with real citation, cap per vtype
    for vtype in VTYPE_PRIORITY:
        for kind, source in (("person", buckets_person.get(vtype, [])),
                             ("company", buckets_company.get(vtype, []))):
            for n, e in source:
                _try_add_law(kind, n, e, vtype, require_cite=True)
                if len(law_pool) >= TARGET_LAW:
                    break
            if len(law_pool) >= TARGET_LAW:
                break
        if len(law_pool) >= TARGET_LAW:
            break

    # Pass 2: drop cap, still prefer real citation
    if len(law_pool) < TARGET_LAW:
        for vtype in VTYPE_PRIORITY:
            for kind, source in (("person", buckets_person.get(vtype, [])),
                                 ("company", buckets_company.get(vtype, []))):
                for n, e in source:
                    if e.get("event_id") in used_event_ids:
                        continue
                    if not cite_re.search(e.get("law") or ""):
                        continue
                    law_pool.append((kind, n, e, vtype))
                    used_event_ids.add(e.get("event_id"))
                    if len(law_pool) >= TARGET_LAW:
                        break
                if len(law_pool) >= TARGET_LAW:
                    break
            if len(law_pool) >= TARGET_LAW:
                break

    # Pass 3: accept fallback laws as last resort
    if len(law_pool) < TARGET_LAW:
        for vtype in VTYPE_PRIORITY:
            for kind, source in (("person", buckets_person.get(vtype, [])),
                                 ("company", buckets_company.get(vtype, []))):
                for n, e in source:
                    if e.get("event_id") in used_event_ids:
                        continue
                    law_pool.append((kind, n, e, vtype))
                    used_event_ids.add(e.get("event_id"))
                    if len(law_pool) >= TARGET_LAW:
                        break
                if len(law_pool) >= TARGET_LAW:
                    break
            if len(law_pool) >= TARGET_LAW:
                break

    law_pool = law_pool[:TARGET_LAW]
    print(f"[pool] case_retrieval={len(case_pool)}  law_grounding={len(law_pool)}")

    # ----------- emit rows -----------
    rows: list[dict] = []
    gid = 51  # gold_051 ...

    def build_row(intent: str, kind: str, name: str, event: dict, vtype: str) -> dict:
        if intent == "case_retrieval":
            q, slots = make_case_query(kind, name, event, vtype)
            kps = make_keypoints_case(name, vtype, event)
            laws = derive_laws(event, vtype)
            notes = (
                f"2026-04-22 / build_gold_extra / single-gold 反向合成 / "
                f"anchor={kind}:{name}"
            )
        else:
            q, slots = make_law_query(kind, name, event, vtype)
            laws = derive_laws(event, vtype)
            kps = make_keypoints_law(name, vtype, event, laws)
            notes = (
                f"2026-04-22 / build_gold_extra / law_grounding single-gold / "
                f"anchor={kind}:{name}"
            )
        return {
            "id": "",
            "intent": intent,
            "query": q,
            "gold_answer_keypoints": kps,
            "relevant_event_ids": [str(event.get("event_id"))],
            "relevant_laws": laws,
            "expected_slots": slots,
            "difficulty": "easy" if intent == "case_retrieval" else "medium",
            "is_trap": False,
            "trap_reason": None,
            "notes": notes,
        }

    # Interleave case / law? Keep it simple: case first, then law, assign IDs.
    for kind, name, ev, vt in case_pool:
        rows.append(build_row("case_retrieval", kind, name, ev, vt))
    for kind, name, ev, vt in law_pool:
        rows.append(build_row("law_grounding", kind, name, ev, vt))

    # Quality gate: query length 10-40 chars; drop violators (shouldn't happen)
    filtered: list[dict] = []
    dropped = 0
    for r in rows:
        q = r["query"]
        if not (10 <= len(q) <= 40):
            dropped += 1
            continue
        # expected_slots must have at least one non-empty value
        if not any(r["expected_slots"].values()):
            dropped += 1
            continue
        filtered.append(r)
    if dropped:
        print(f"[warn] dropped {dropped} rows failing length/slots gate")

    filtered = filtered[:TARGET_TOTAL]
    for idx, r in enumerate(filtered):
        r["id"] = f"gold_{gid + idx:03d}"

    # Write out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    ct_intent = collections.Counter(r["intent"] for r in filtered)
    ct_vt = collections.Counter(r["expected_slots"].get("violation_type", "?") for r in filtered)
    print(f"[write] {OUT_PATH} rows={len(filtered)}")
    print(f"[dist] intent={dict(ct_intent)}")
    print(f"[dist] violation_type={dict(ct_vt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_gold_100.py

Merge ``data/eval/gold_50.jsonl`` + ``data/eval/gold_extra_candidates.jsonl``
into ``data/eval/gold_100.jsonl``. The original ``gold_50.jsonl`` is NOT
modified.

Self-validation performed before writing:

* No duplicate ``id`` across the combined set.
* Every ``relevant_event_ids`` (if any) exists in
  ``data/processed/event_corpus.jsonl``.
* For **extra** rows (single-gold, non-trap): the one anchor keyword in the
  query must literally appear in the referenced event's ``activity`` field,
  guaranteeing the gold event is retrievable via BM25 / dense search.
* Schema sanity: required fields present, ``relevant_event_ids`` length == 1
  for every extra row, ``is_trap`` is False, ``difficulty`` ∈ {easy, medium}.

Stdlib-only.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "eval" / "gold_50.jsonl"
EXTRA_PATH = ROOT / "data" / "eval" / "gold_extra_candidates.jsonl"
OUT_PATH = ROOT / "data" / "eval" / "gold_100.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "event_corpus.jsonl"

REQUIRED_FIELDS = (
    "id", "intent", "query", "gold_answer_keypoints",
    "relevant_event_ids", "relevant_laws", "expected_slots",
    "difficulty", "is_trap", "notes",
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[{path}] line {ln} JSON 解析失败: {exc}")
    return rows


def load_events(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            out[str(e.get("event_id"))] = e
    return out


KNOWN_VTYPE_LABELS = (
    "内幕交易", "虚假记载", "推迟披露", "违规买卖股票", "虚构利润",
    "操纵股价", "占用公司资产", "违规担保", "重大遗漏",
    "一般会计处理不当", "披露不实", "欺诈上市",
)


def extract_query_anchor(query: str) -> str:
    """Return the company/person anchor substring from an extra-row query.

    Extra rows are generated as "<NAME>（<role>）因..." or "<NAME>因..." or
    "<NAME><vtype>案援引了...". We strip the suffix to get NAME.
    """
    q = query
    # drop parenthetical role before matching "因"
    q = re.sub(r"（[^）]*）", "", q)
    # case_retrieval: "<NAME>因..."
    idx = q.find("因")
    if idx > 0:
        return q[:idx]
    # law_grounding: "<NAME><vtype>案援引..."
    idx = q.find("案援引")
    if idx > 0:
        head = q[:idx]
        for vt in KNOWN_VTYPE_LABELS:
            if head.endswith(vt):
                return head[: -len(vt)]
        return head
    return q


def check(errors: list[str], cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def main() -> int:
    errors: list[str] = []
    originals = load_jsonl(GOLD_PATH)
    extras = load_jsonl(EXTRA_PATH)
    corpus = load_events(CORPUS_PATH)

    print(f"[load] originals={len(originals)} extras={len(extras)} corpus={len(corpus)}")

    # 1) dedup id
    all_ids = [r.get("id") for r in originals] + [r.get("id") for r in extras]
    counts = collections.Counter(all_ids)
    dupes = [k for k, v in counts.items() if v > 1]
    check(errors, not dupes, f"ID 冲突: {dupes}")

    # 2) event_id existence
    for r in originals + extras:
        for eid in r.get("relevant_event_ids") or []:
            if str(eid) not in corpus:
                errors.append(f"[{r.get('id')}] event_id {eid} 不在 event_corpus")

    # 3) extras-only strict checks
    for r in extras:
        rid = r.get("id", "?")
        for fld in REQUIRED_FIELDS:
            if fld not in r:
                errors.append(f"[{rid}] 缺少字段 {fld}")
        check(errors, r.get("intent") in {"case_retrieval", "law_grounding"},
              f"[{rid}] intent 必须为 case_retrieval / law_grounding")
        check(errors, r.get("is_trap") is False, f"[{rid}] is_trap 必须为 False")
        check(errors, r.get("difficulty") in {"easy", "medium"},
              f"[{rid}] difficulty 必须为 easy / medium")
        ev_ids = r.get("relevant_event_ids") or []
        check(errors, len(ev_ids) == 1,
              f"[{rid}] 单 gold 规格要求 relevant_event_ids 长度为 1，实际 {len(ev_ids)}")
        q = r.get("query") or ""
        check(errors, 10 <= len(q) <= 40,
              f"[{rid}] query 字符数必须在 10-40，实际 {len(q)}")
        slots = r.get("expected_slots") or {}
        check(errors, any(v not in (None, "", [], {}) for v in slots.values()),
              f"[{rid}] expected_slots 至少需 1 个非空 value")
        kps = r.get("gold_answer_keypoints") or []
        check(errors, 2 <= len(kps) <= 3,
              f"[{rid}] gold_answer_keypoints 数量应 ∈ [2,3]，实际 {len(kps)}")
        laws = r.get("relevant_laws") or []
        check(errors, 1 <= len(laws) <= 2,
              f"[{rid}] relevant_laws 数量应 ∈ [1,2]，实际 {len(laws)}")
        # anchor in activity
        if ev_ids:
            ev = corpus.get(str(ev_ids[0])) or {}
            activity = ev.get("activity") or ""
            anchor = extract_query_anchor(q)
            if anchor:
                if anchor not in activity:
                    # allow short prefix match (trim "股份有限公司" etc.)
                    short = re.sub(r"(股份)?(有限)?(责任)?公司$", "", anchor)
                    if len(short) < 4 or short not in activity:
                        errors.append(
                            f"[{rid}] query 锚点 '{anchor}' 未出现在 event "
                            f"{ev_ids[0]} 的 activity 中"
                        )

    if errors:
        print("[FAIL] 校验失败，终止写入：")
        for e in errors[:40]:
            print(f"  - {e}")
        if len(errors) > 40:
            print(f"  ... 共 {len(errors)} 条问题")
        return 1

    # Write merged file (originals first in natural order, then extras)
    merged = list(originals) + list(extras)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final report
    intent_ct = collections.Counter(r.get("intent", "?") for r in merged)
    trap_ct = sum(1 for r in merged if r.get("is_trap"))
    eligible_intents = {
        "case_retrieval", "law_grounding",
        "sanction_recommendation", "trend_analysis",
    }
    eligible = [r for r in merged
                if r.get("intent") in eligible_intents
                and not r.get("is_trap")
                and (r.get("relevant_event_ids") or [])]
    single_gold = sum(1 for r in eligible if len(r["relevant_event_ids"]) == 1)
    multi_gold = sum(1 for r in eligible if len(r["relevant_event_ids"]) > 1)

    print(f"[write] {OUT_PATH} total={len(merged)}")
    print(f"[dist] intent={dict(intent_ct)}")
    print(f"[dist] traps={trap_ct}")
    print(f"[dist] eligible(retrieval-eligible)={len(eligible)}")
    print(f"[dist]   single_gold={single_gold}  multi_gold={multi_gold}")
    print("[PASS] merge + validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_gold_50.py

校验 `data/eval/gold_50.jsonl` 是否符合 M3b 金标集规范：
  1. 总条数 == 50
  2. 5 类 intent 分布严格匹配：
       case_retrieval          : 15
       law_grounding           : 10
       sanction_recommendation : 10
       trend_analysis          : 10
       边界（out_of_scope / multi_turn_followup，含 1 条 hallucination trap）: 5
  3. 每条 relevant_event_ids 中的 EventID 都必须在 event_corpus.jsonl 中真实存在（陷阱题允许空）
  4. 陷阱题（is_trap == True）必须标注 trap_reason，且 relevant_event_ids 为空
  5. 每条 gold_answer_keypoints 至少 2 条，且含硬性限定词（"必须"/"不得"/"不应"/"允许"）
  6. 核心幻觉陷阱的公司名不得出现在任何事件的 parties/title/retrieval_text 中

运行：
  python scripts/validate_gold_50.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "eval" / "gold_50.jsonl"
CORPUS_PATH = ROOT / "data" / "processed" / "event_corpus.jsonl"

EXPECTED_TOTAL = 50
EXPECTED_INTENT_DIST = {
    "case_retrieval": 15,
    "law_grounding": 10,
    "sanction_recommendation": 10,
    "trend_analysis": 10,
}
BOUNDARY_INTENTS = {"out_of_scope", "multi_turn_followup"}
EXPECTED_BOUNDARY = 5
HARD_KEYWORDS = ("必须", "不得", "不应", "允许", "应当")


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


def load_corpus(path: Path) -> tuple[set[str], list[dict]]:
    ids: set[str] = set()
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            ids.add(str(e.get("event_id")))
            events.append(e)
    return ids, events


def check(errors: list[str], cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def main() -> int:
    errors: list[str] = []

    if not GOLD_PATH.exists():
        raise SystemExit(f"[FAIL] 未找到 {GOLD_PATH}")
    if not CORPUS_PATH.exists():
        raise SystemExit(f"[FAIL] 未找到 {CORPUS_PATH}")

    rows = load_jsonl(GOLD_PATH)
    corpus_ids, corpus_events = load_corpus(CORPUS_PATH)

    # Rule 1: total count
    check(errors, len(rows) == EXPECTED_TOTAL,
          f"总条数应为 {EXPECTED_TOTAL}，实际 {len(rows)}")

    # Rule 2: intent distribution
    # 边界桶 = out_of_scope + multi_turn_followup + 任意 is_trap=True 的条目
    # 四个 in-scope 类计数时排除 is_trap=True 的条目
    intent_ct = Counter(r.get("intent", "") for r in rows)
    inscope_ct = Counter(r.get("intent", "") for r in rows if not r.get("is_trap"))
    boundary_ct = sum(1 for r in rows
                       if (r.get("intent") in BOUNDARY_INTENTS) or r.get("is_trap"))
    for intent, expected in EXPECTED_INTENT_DIST.items():
        check(errors, inscope_ct.get(intent, 0) == expected,
              f"非陷阱 intent={intent} 应有 {expected} 条，实际 {inscope_ct.get(intent, 0)} 条")
    check(errors, boundary_ct == EXPECTED_BOUNDARY,
          f"边界桶（out_of_scope + multi_turn_followup + is_trap）应为 {EXPECTED_BOUNDARY} 条，实际 {boundary_ct} 条")

    # Rule 3: relevant_event_ids 必须真实存在（trap 允许空；trend_analysis 允许空但若给则必须真实）
    missing_ids: list[tuple[str, str]] = []
    for r in rows:
        rid = r.get("id", "?")
        for eid in r.get("relevant_event_ids") or []:
            if str(eid) not in corpus_ids:
                missing_ids.append((rid, str(eid)))
    check(errors, not missing_ids,
          "存在 relevant_event_ids 不在 event_corpus 中: " + ", ".join(f"{a}->{b}" for a, b in missing_ids))

    # Rule 4: traps
    trap_rows = [r for r in rows if r.get("is_trap")]
    check(errors, len(trap_rows) >= 1, "至少需要 1 条 is_trap=true 的陷阱样本")
    for r in trap_rows:
        rid = r.get("id", "?")
        check(errors, bool(r.get("trap_reason")), f"[{rid}] 陷阱样本必须填写 trap_reason")
        check(errors, not (r.get("relevant_event_ids") or []),
              f"[{rid}] 陷阱样本的 relevant_event_ids 必须为空")

    # Rule 5: keypoint schema (>=2 + 硬性限定词)
    for r in rows:
        rid = r.get("id", "?")
        kps = r.get("gold_answer_keypoints") or []
        check(errors, len(kps) >= 2, f"[{rid}] gold_answer_keypoints 数量应 ≥ 2，实际 {len(kps)}")
        for i, kp in enumerate(kps):
            if not any(kw in kp for kw in HARD_KEYWORDS):
                errors.append(f"[{rid}] keypoint#{i} 缺少硬性限定词({HARD_KEYWORDS}): {kp!r}")

    # Rule 5b: required fields
    required_fields = ("id", "intent", "query", "gold_answer_keypoints",
                       "relevant_event_ids", "relevant_laws", "expected_slots",
                       "difficulty", "is_trap", "notes")
    for r in rows:
        rid = r.get("id", "?")
        for fld in required_fields:
            if fld not in r:
                errors.append(f"[{rid}] 缺少必需字段 {fld}")

    # Rule 5c: id 唯一
    ids_seen = Counter(r.get("id") for r in rows)
    dupes = [k for k, v in ids_seen.items() if v > 1]
    check(errors, not dupes, f"重复 id: {dupes}")

    # Rule 6: hallucination probe 公司名在 corpus 中真不存在
    fake_name = "华夏腾飞智能科技股份有限公司"
    fake_hits = []
    for e in corpus_events:
        blob = (e.get("title") or "") + "||" + "|".join(e.get("parties") or []) + "||" + (e.get("retrieval_text") or "")
        if fake_name in blob or "华夏腾飞" in blob:
            fake_hits.append(e.get("event_id"))
    check(errors, not fake_hits,
          f"核心幻觉陷阱公司名 '{fake_name}' 竟然出现在 corpus 中：{fake_hits[:5]}（需更换陷阱公司名）")

    # --- 报告 ---
    print("=" * 60)
    print("gold_50.jsonl 校验报告")
    print("=" * 60)
    print(f"总条数: {len(rows)}")
    print("intent 分布（含陷阱）:")
    for k in sorted(intent_ct):
        print(f"  - {k}: {intent_ct[k]}")
    print("in-scope 分布（不含陷阱）:")
    for k in sorted(inscope_ct):
        print(f"  - {k}: {inscope_ct[k]}")
    print(f"边界桶合计: {boundary_ct}")
    print(f"陷阱题数: {len(trap_rows)}")
    print(f"hallucination probe 公司名在 corpus 命中数: {len(fake_hits)}（应为 0）")
    all_ref_ids = {eid for r in rows for eid in (r.get("relevant_event_ids") or [])}
    print(f"引用的 distinct EventID 总数: {len(all_ref_ids)}")
    print()

    if errors:
        print("[FAIL] 发现以下问题：")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("[PASS] 所有校验项通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build static JSON data for docs/showcase/*.html (GitHub Pages static demo).

Reads pipeline reports and produces small, self-contained JSON files that the
static HTML pages can fetch. Everything is pure read + JSON write - no models.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "reports"
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "docs" / "showcase" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_gold_index() -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    path = EVAL / "gold_130.jsonl"
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mapping[row["id"]] = row
    return mapping


def build_compare() -> dict:
    """Build G0 vs G3 per-sample comparison for the interactive HTML."""
    m44 = _load_json(REPORTS / "m4_4_generation_eval.json")
    if m44 is None:
        raise SystemExit("docs/reports/m4_4_generation_eval.json not found")
    gold_idx = _load_gold_index()
    det = m44["details"]

    # Group rows by gold_id for easy alignment across 4 conditions
    by_id: dict[str, dict] = {}
    for cond in ("G0", "G1", "G2", "G3"):
        for row in det[cond]:
            gid = row["gold_id"]
            entry = by_id.setdefault(
                gid,
                {
                    "gold_id": gid,
                    "query": row["query"],
                    "gold_info": {},
                    "conditions": {},
                },
            )
            entry["conditions"][cond] = {
                "answer": row["answer"],
                "cited_event_ids": row.get("cited_event_ids", []),
                "event_id_hit": row.get("event_id_hit", False),
                "format_ok": row.get("format_ok", False),
                "hallucinated_numbers": row.get("hallucinated_numbers", []),
                "answer_len": row.get("answer_len", 0),
                "latency_s": row.get("latency_s", 0.0),
                "retrieved_event_ids": row.get("retrieved_event_ids", []),
            }

    # Enrich with gold metadata
    for gid, entry in by_id.items():
        g = gold_idx.get(gid)
        if g:
            entry["gold_info"] = {
                "intent": g.get("intent"),
                "difficulty": g.get("difficulty"),
                "relevant_event_ids": g.get("relevant_event_ids", []),
                "relevant_laws": g.get("relevant_laws", []),
                "gold_answer_keypoints": g.get("gold_answer_keypoints", []),
                "notes": g.get("notes", ""),
            }

    samples = list(by_id.values())
    # Sort: showcase strongest wins first (G3 correct, G0 not)
    def _score(s: dict) -> int:
        g0 = s["conditions"].get("G0", {})
        g3 = s["conditions"].get("G3", {})
        # prefer: G3 hits + G0 hallucinates
        score = 0
        if g3.get("event_id_hit") and not g0.get("event_id_hit"):
            score += 10
        if g3.get("format_ok") and not g0.get("format_ok"):
            score += 5
        if g0.get("hallucinated_numbers") and not g3.get("hallucinated_numbers"):
            score += 3
        return score

    samples.sort(key=_score, reverse=True)

    summary = m44["summary"]
    return {
        "summary": summary,
        "samples": samples,
        "meta": {
            "n_samples": len(samples),
            "conditions": {
                "G0": "base Qwen-0.5B, no RAG, weak prompt",
                "G1": "base Qwen-0.5B + RAG, weak prompt",
                "G2": "base Qwen-0.5B + RAG + strong prompt",
                "G3": "Qwen-0.5B + LoRA + RAG + strong prompt (本项目)",
            },
            "metric_defs": {
                "event_id_hit": "引用的 EventID 与 gold 集合有交集",
                "format_ok": "答案中至少出现一个 [EventID=xxx]",
                "hallucinated_numbers": "答案中含证据里不存在的金额/百分比数字",
            },
        },
    }


def build_retrieval() -> dict | None:
    """Build retrieval ablation summary for the showcase page."""
    candidates = [
        REPORTS / "m3e_retrieval_report.json",
        REPORTS / "m3_retrieval_report.json",
    ]
    for p in candidates:
        data = _load_json(p)
        if data is not None:
            return data
    return None


def build_trend() -> dict | None:
    return _load_json(REPORTS / "m4_2_trend_eval.json")


def build_macbert() -> dict | None:
    return _load_json(REPORTS / "m5_macbert_report.json")


def main() -> None:
    compare = build_compare()
    (OUT / "compare.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[compare] wrote {OUT / 'compare.json'} ({len(compare['samples'])} samples)")

    retrieval = build_retrieval()
    if retrieval is not None:
        (OUT / "retrieval.json").write_text(
            json.dumps(retrieval, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[retrieval] wrote {OUT / 'retrieval.json'}")

    trend = build_trend()
    if trend is not None:
        (OUT / "trend.json").write_text(
            json.dumps(trend, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[trend] wrote {OUT / 'trend.json'}")

    macbert = build_macbert()
    if macbert is not None:
        (OUT / "macbert.json").write_text(
            json.dumps(macbert, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[macbert] wrote {OUT / 'macbert.json'}")


if __name__ == "__main__":
    main()

"""Smoke-test only: build 50 minimal QA samples from event_corpus for M1 QLoRA smoke.

This is throwaway data purely for link-smoke (forward/backward). Not for real training.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "event_corpus.jsonl"
OUT_DIR = ROOT / "data" / "train_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "smoke_qa.jsonl"

N = 50


def trunc(s: str, n: int = 120) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    return s[:n]


def build():
    rows = []
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= N:
                break

    out = []
    for ev in rows:
        eid = ev.get("event_id", "?")
        act = trunc(ev.get("activity", ""), 120)
        vtypes = "、".join(ev.get("violation_types") or [])
        q = f"列出一个{vtypes or '违规'}的真实案例"
        a = f"根据 [EventID={eid}] 记录：{act or '相关违规行为'}"
        out.append({
            "instruction": q,
            "input": "",
            "output": a,
        })

    with OUT.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} samples -> {OUT}")


if __name__ == "__main__":
    build()

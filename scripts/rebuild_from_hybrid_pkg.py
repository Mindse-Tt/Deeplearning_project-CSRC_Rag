"""
Rebuild event_corpus.jsonl, event_chunks.jsonl, and chunk_id_order.json
from the hybrid_rag_data_package provided by the teammate.

Run from the project root:
    .venv311/Scripts/python scripts/rebuild_from_hybrid_pkg.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from csrc_rag.data.builders import build_event_corpus, write_jsonl  # noqa: E402
from csrc_rag.settings import PROCESSED_DIR  # noqa: E402

# 队友提供的 hybrid 数据包路径：可用环境变量 HYBRID_PKG 覆盖，默认取项目内 data/hybrid_rag_data_package
HYBRID_PKG = Path(os.environ.get("HYBRID_PKG", PROJECT_ROOT / "data" / "hybrid_rag_data_package"))


def _clean_nan(value):
    """Convert pandas NaN / float NaN to None."""
    if value is None:
        return None
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    return value


def _safe_str(value) -> str | None:
    v = _clean_nan(value)
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() not in {"nan", "none", ""} else None


def _safe_float(value) -> float | None:
    v = _clean_nan(value)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_chunks_jsonl(chunks_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for i, row in enumerate(chunks_df.itertuples(index=False)):
        chunk_id = f"c{i:06d}"
        event_id = _safe_str(getattr(row, "EventID", None)) or "UNKNOWN"
        chunk_text = _safe_str(getattr(row, "chunk_text", None)) or ""
        full_name = _safe_str(getattr(row, "FullName", None)) or ""
        violation_type = _safe_str(getattr(row, "ViolationType", None)) or ""
        punishment_type = _safe_str(getattr(row, "PunishmentType", None)) or ""
        promulgator = _safe_str(getattr(row, "Promulgator", None)) or ""
        declare_date = _safe_str(getattr(row, "DeclareDate", None)) or ""
        symbol = _safe_str(getattr(row, "Symbol", None)) or ""
        is_listed = _safe_str(getattr(row, "IsListedCom", None))
        sum_penalty = _safe_float(getattr(row, "SumPenalty", None))
        chunk_type = _safe_str(getattr(row, "chunk_type", None)) or ""
        source_col = _safe_str(getattr(row, "source_col", None)) or ""

        # BM25 retrieval text: combine content with metadata keywords
        parts = [p for p in [chunk_text, violation_type, punishment_type, promulgator, full_name] if p]
        retrieval_text = " ".join(parts)

        rows.append(
            {
                "chunk_id": chunk_id,
                "event_id": event_id,
                "full_name": full_name,
                "symbol": symbol,
                "is_listed_company": is_listed,
                "declare_date": declare_date,
                "promulgator": promulgator,
                "violation_type": violation_type,
                "punishment_type": punishment_type,
                "sum_penalty": sum_penalty,
                "chunk_type": chunk_type,
                "source_col": source_col,
                "chunk_text": chunk_text,
                "retrieval_text": retrieval_text,
                "year": declare_date[:4] if len(declare_date) >= 4 and declare_date[:4].isdigit() else None,
            }
        )
    return rows


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a pandas DataFrame to list[dict] with NaN cleaned to None."""
    result = []
    for row in df.itertuples(index=False):
        record: dict = {}
        for col in df.columns:
            val = getattr(row, col, None)
            record[col] = _clean_nan(val)
        result.append(record)
    return result


def main() -> None:
    print(f"Hybrid package path: {HYBRID_PKG}")
    if not HYBRID_PKG.exists():
        print("ERROR: Hybrid package directory not found. Check the path.")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load source files ──────────────────────────────────────────────────
    print("Loading records_df.xlsx …")
    records_df = pd.read_excel(HYBRID_PKG / "records_df.xlsx", dtype=str)
    print(f"  records_df: {len(records_df)} rows × {len(records_df.columns)} cols")

    print("Loading chunks_df.xlsx …")
    chunks_df = pd.read_excel(HYBRID_PKG / "chunks_df.xlsx", dtype=str)
    # SumPenalty needs numeric — re-read as mixed
    chunks_df_num = pd.read_excel(HYBRID_PKG / "chunks_df.xlsx")
    chunks_df["SumPenalty"] = chunks_df_num["SumPenalty"] if "SumPenalty" in chunks_df_num.columns else None
    print(f"  chunks_df:  {len(chunks_df)} rows × {len(chunks_df.columns)} cols")

    # SumPenalty numeric for records too
    records_df_num = pd.read_excel(HYBRID_PKG / "records_df.xlsx")
    if "SumPenalty" in records_df_num.columns:
        records_df["SumPenalty"] = records_df_num["SumPenalty"]

    # ── 2. Build event_corpus.jsonl ───────────────────────────────────────────
    print("Building event_corpus.jsonl …")
    records_list = df_to_records(records_df)
    corpus = build_event_corpus(records_list)
    out_corpus = PROCESSED_DIR / "event_corpus.jsonl"
    write_jsonl(out_corpus, corpus)
    print(f"  Wrote {len(corpus)} events → {out_corpus}")

    # ── 3. Build event_chunks.jsonl + chunk_id_order.json ─────────────────────
    print("Building event_chunks.jsonl …")
    chunks = build_chunks_jsonl(chunks_df)
    out_chunks = PROCESSED_DIR / "event_chunks.jsonl"
    write_jsonl(out_chunks, chunks)
    print(f"  Wrote {len(chunks)} chunks → {out_chunks}")

    chunk_id_order = [c["chunk_id"] for c in chunks]
    out_order = PROCESSED_DIR / "chunk_id_order.json"
    with out_order.open("w", encoding="utf-8") as fh:
        json.dump(chunk_id_order, fh, ensure_ascii=False)
    print(f"  Wrote chunk_id_order.json ({len(chunk_id_order)} ids) → {out_order}")

    # ── 4. Copy chunk_embeddings.npy ─────────────────────────────────────────
    src_npy = HYBRID_PKG / "chunk_embeddings.npy"
    dst_npy = PROCESSED_DIR / "chunk_embeddings.npy"
    if src_npy.exists():
        print(f"Copying chunk_embeddings.npy ({src_npy.stat().st_size // 1024 // 1024} MB) …")
        shutil.copy2(src_npy, dst_npy)
        print(f"  Copied → {dst_npy}")
    else:
        print(f"WARNING: {src_npy} not found. Dense retrieval will be unavailable.")

    print("\n✓ Data rebuild complete.")
    print(f"  event_corpus.jsonl : {len(corpus)} events")
    print(f"  event_chunks.jsonl : {len(chunks)} chunks")
    print(f"  chunk_id_order.json: {len(chunk_id_order)} ids")
    print(f"  chunk_embeddings.npy: {'copied' if dst_npy.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()

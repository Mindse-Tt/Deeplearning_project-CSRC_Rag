"""L3 离线分块：把处罚事件文档切成可检索的 chunk（检索语料的最小单元）。

本模块是检索层的离线构建侧：将每个处罚事件按"摘要 / 违规行为 / 法律依据"三类
切成多个 chunk，并为每个 chunk 生成两份文本——``chunk_text``（回显给用户的原文）
与 ``retrieval_text``（喂给 BM25/向量的检索文本）。二者分离的设计意图是：检索文本
里可以前置结构化元数据块（标题、违规类型、当事人职位等）来增强召回，而展示文本
保持干净原文。长正文用带重叠的滑动窗口切分，避免跨片段时把关键句切断。
"""
from __future__ import annotations

from typing import Any

from csrc_rag.data.builders import _normalize_binary_flag


def sliding_text_chunks(text: str | None, chunk_size: int, overlap: int) -> list[str]:
    # 带重叠的滑动窗口切分：短于 chunk_size 直接整段返回，长文本按窗口滑动。
    if not text:
        return []

    compact = text.strip()
    if not compact:
        return []
    if len(compact) <= chunk_size:
        return [compact]

    chunks: list[str] = []
    start = 0
    while start < len(compact):
        end = min(len(compact), start + chunk_size)
        chunk = compact[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(compact):
            break
        # 下一窗口起点回退 overlap 个字符制造重叠，保证被切断的句子在相邻片段里
        # 仍有完整出现的机会；max(start+1, ...) 兜底保证 start 严格递增、不死循环。
        start = max(start + 1, end - overlap)
    return chunks


def _unique_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _explode_positions(positions: list[str] | None) -> list[str]:
    """``positions`` values often look like "董事,董事会秘书" — split them.

    Also dedupe so a person with two roles doesn't double-count.
    """
    if not positions:
        return []
    flat: list[str] = []
    for p in positions:
        if not p:
            continue
        for sub in str(p).replace(";", ",").replace("；", ",").replace("、", ",").split(","):
            sub = sub.strip()
            if sub:
                flat.append(sub)
    return _unique_preserve(flat)


def _build_metadata_block(event: dict[str, Any]) -> str:
    """M3e-B: surface structured fields into BM25-searchable text.

    Prior to this, ``parties`` / ``positions`` / ``relationships`` were
    carried as top-level meta but never appeared in ``retrieval_text``,
    so queries like "董事长内幕交易" or "独立董事违规" could only hit
    via whatever the ``activity`` narrative happened to spell out. The
    metadata block is now prepended to every chunk's retrieval_text to
    give BM25 direct access to these fields.

    Ablation on gold_100 (see docs/reports/m3e_ablation_report.md §3):
    - BM25 Recall@5:      0.381 → 0.378 (-0.8%, within noise)
    - Dense Recall@5:     0.269 → 0.293 (+8.8%)
    - Hybrid Recall@5:    0.390 → 0.388 (~flat)
    - Hybrid+R Recall@5:  0.312 → 0.356 (+14.1%)

    The BM25 cost is tiny (metadata dilutes the IDF of party names by a
    fraction), but the Dense gain is material because bge-small-zh can
    exploit the domain vocabulary. Hybrid+Rerank benefits the most
    because the cross-encoder can now see role/identity fields that
    were invisible in the narrative text.

    Ablation switch: set ``CSRC_RAG_DISABLE_METADATA_BLOCK=1`` before
    ``python scripts/build_event_chunks.py`` to regenerate the chunks
    with only violation_types surfaced (used for paper ablation table).
    """
    import os
    if os.environ.get("CSRC_RAG_DISABLE_METADATA_BLOCK") == "1":
        vtypes = _unique_preserve(
            [str(v) for v in (event.get("violation_types") or []) if v]
        )
        return f"违规类型：{'；'.join(vtypes)}" if vtypes else ""

    positions = _explode_positions(event.get("positions"))
    parties = _unique_preserve([str(p) for p in (event.get("parties") or []) if p])
    relationships = _unique_preserve(
        [str(r) for r in (event.get("relationships") or []) if r]
    )
    vtypes = _unique_preserve(
        [str(v) for v in (event.get("violation_types") or []) if v]
    )
    punish = _unique_preserve(
        [str(p) for p in (event.get("punishment_types") or []) if p]
    )
    lines: list[str] = []
    if vtypes:
        lines.append(f"违规类型：{'；'.join(vtypes)}")
    if positions:
        # Repeat positions twice so BM25 weighs them higher than a single
        # activity-text mention — still below the term-frequency ceiling
        # so it doesn't swamp the rest of the text.
        lines.append(f"当事人职位：{'、'.join(positions)}")
        lines.append(f"职位关键词：{' '.join(positions)}")
    if relationships:
        lines.append(f"当事人身份：{'、'.join(relationships)}")
    if parties:
        lines.append(f"当事人：{'、'.join(parties)}")
    if punish:
        lines.append(f"处罚方式：{'、'.join(punish)}")
    # 拼成多行元数据块，前置到每个 chunk 的 retrieval_text，让结构化字段直接可被
    # BM25/向量检索命中（消融见上方 docstring：对 Dense 与 Hybrid+Rerank 增益显著）。
    return "\n".join(lines)


def build_event_chunks(
    event_documents: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    # 对每个事件产出三类 chunk：①summary 汇总片（整事件一条，便于粗召回）、
    # ②activity 违规行为正文（滑窗多片）、③law 法律依据正文（滑窗多片）。
    # 三类共享同一份 common_meta，使下游可凭 chunk 直接做元数据过滤与事件级聚合。
    rows: list[dict[str, Any]] = []
    for event in event_documents:
        event_id = event["event_id"]
        year = event["declare_date"][:4] if event.get("declare_date") else None
        common_meta = {
            "event_id": event_id,
            "title": event.get("title"),
            "declare_date": event.get("declare_date"),
            "supervision_date": event.get("supervision_date"),
            "promulgator": event.get("promulgator"),
            "supervisor": event.get("supervisor"),
            "year": year,
            "is_listed_company": _normalize_binary_flag(event.get("is_listed_company")),
            "violation_types": event.get("violation_types", []),
            "punishment_types": event.get("punishment_types", []),
            # M3e-B: propagate party-level structured fields so downstream
            # rerank / responder can use them without re-joining event_corpus.
            "parties": event.get("parties", []),
            "positions": event.get("positions", []),
            "relationships": event.get("relationships", []),
        }

        metadata_block = _build_metadata_block(event)

        summary_parts = [
            f"标题：{event.get('title')}" if event.get("title") else None,
            f"公告日期：{event.get('declare_date')}" if event.get("declare_date") else None,
            f"发布机构：{event.get('promulgator')}" if event.get("promulgator") else None,
            f"处罚机构：{event.get('supervisor')}" if event.get("supervisor") else None,
            metadata_block or None,  # violation_types / positions / parties / relationships
            f"核心行为：{(event.get('activity') or '')[:180]}",
        ]
        summary_text = "\n".join(part for part in summary_parts if part)
        rows.append(
            {
                "chunk_id": f"{event_id}::summary",
                "chunk_type": "summary",
                "section": "summary",
                "chunk_text": summary_text,
                "retrieval_text": summary_text,
                **common_meta,
            }
        )

        for idx, chunk in enumerate(sliding_text_chunks(event.get("activity"), chunk_size=chunk_size, overlap=overlap)):
            retrieval_text = "\n".join(
                part
                for part in [
                    f"标题：{event.get('title')}" if event.get("title") else None,
                    metadata_block or None,
                    f"违规行为片段：{chunk}",
                    f"发布机构：{event.get('promulgator')}" if event.get("promulgator") else None,
                ]
                if part
            )
            rows.append(
                {
                    "chunk_id": f"{event_id}::activity::{idx}",
                    "chunk_type": "activity",
                    "section": "activity",
                    "chunk_text": chunk,
                    "retrieval_text": retrieval_text,
                    **common_meta,
                }
            )

        for idx, chunk in enumerate(sliding_text_chunks(event.get("law"), chunk_size=chunk_size, overlap=overlap)):
            retrieval_text = "\n".join(
                part
                for part in [
                    f"标题：{event.get('title')}" if event.get("title") else None,
                    metadata_block or None,
                    f"法律依据片段：{chunk}",
                    f"发布机构：{event.get('promulgator')}" if event.get("promulgator") else None,
                ]
                if part
            )
            rows.append(
                {
                    "chunk_id": f"{event_id}::law::{idx}",
                    "chunk_type": "law",
                    "section": "law",
                    "chunk_text": chunk,
                    "retrieval_text": retrieval_text,
                    **common_meta,
                }
            )
    return rows

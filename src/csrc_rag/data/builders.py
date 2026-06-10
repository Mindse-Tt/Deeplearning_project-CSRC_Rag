from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from csrc_rag.data.schema import normalize_text, split_multilabel


def _normalize_binary_flag(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return "1" if float(value) >= 1 else "0"
    except (TypeError, ValueError):
        text = normalize_text(value)
        if text in {"1", "true", "True", "是"}:
            return "1"
        if text in {"0", "false", "False", "否"}:
            return "0"
        return text


def _first_non_empty(records: list[dict[str, Any]], field: str) -> Any:
    for record in records:
        value = record.get(field)
        if value not in (None, ""):
            return value
    return None


def _unique_non_empty(records: list[dict[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for record in records:
        value = normalize_text(record.get(field))
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _join_sections(sections: list[tuple[str, str | None]]) -> str:
    lines: list[str] = []
    for title, content in sections:
        if content:
            lines.append(f"{title}：{content}")
    return "\n".join(lines)


def build_event_corpus(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        event_id = str(record.get("EventID"))
        grouped[event_id].append(record)

    corpus: list[dict[str, Any]] = []
    for event_id, group in grouped.items():
        activity = normalize_text(_first_non_empty(group, "Activity"))
        law = normalize_text(_first_non_empty(group, "Law"))
        file_name = normalize_text(_first_non_empty(group, "FileName"))
        promulgator = normalize_text(_first_non_empty(group, "Promulgator"))
        supervisor = normalize_text(_first_non_empty(group, "Supervisor"))
        declare_date = normalize_text(_first_non_empty(group, "DeclareDate"))
        supervision_date = normalize_text(_first_non_empty(group, "SupervisionDate"))
        is_listed_company = _normalize_binary_flag(_first_non_empty(group, "IsListedCom"))
        symbols = _unique_non_empty(group, "Symbol")
        violation_types = _unique_non_empty(group, "ViolationType")
        parties = _unique_non_empty(group, "Party")
        positions = _unique_non_empty(group, "Position")
        relationships = _unique_non_empty(group, "Relationship")
        punishment_types = sorted({label for record in group for label in split_multilabel(record.get("PunishmentType"))})
        punishment_measures = _unique_non_empty(group, "PunishmentMeasure")
        penalties = [record.get("SumPenalty") for record in group if record.get("SumPenalty") not in (None, "")]

        retrieval_text = _join_sections(
            [
                ("标题", file_name),
                ("违规行为", activity),
                ("法律依据", law),
                ("发布机构", promulgator),
                ("处罚机构", supervisor),
                ("公告日期", declare_date),
                ("处罚日期", supervision_date),
                ("违规类型", "；".join(violation_types) if violation_types else None),
                ("涉及主体", "；".join(parties[:20]) if parties else None),
                ("主体职位", "；".join(positions[:20]) if positions else None),
            ]
        )
        reference_text = _join_sections(
            [
                ("处罚方式", "；".join(punishment_types) if punishment_types else None),
                ("处分措施", "；".join(punishment_measures[:10]) if punishment_measures else None),
                ("处罚金额", str(sum(float(x) for x in penalties)) if penalties else None),
            ]
        )

        corpus.append(
            {
                "event_id": event_id,
                "row_count": len(group),
                "title": file_name,
                "declare_date": declare_date,
                "supervision_date": supervision_date,
                "is_listed_company": is_listed_company,
                "symbols": symbols,
                "promulgator": promulgator,
                "supervisor": supervisor,
                "activity": activity,
                "law": law,
                "violation_types": violation_types,
                "parties": parties,
                "positions": positions,
                "relationships": relationships,
                "punishment_types": punishment_types,
                "retrieval_text": retrieval_text,
                "reference_text": reference_text,
            }
        )
    return sorted(corpus, key=lambda item: item["event_id"])


def build_party_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        event_id = str(record.get("EventID"))
        number = record.get("Number")
        party = normalize_text(record.get("Party"))
        position = normalize_text(record.get("Position"))
        relationship = normalize_text(record.get("Relationship"))
        activity = normalize_text(record.get("Activity"))
        law = normalize_text(record.get("Law"))
        promulgator = normalize_text(record.get("Promulgator"))
        declare_date = normalize_text(record.get("DeclareDate"))
        is_listed = record.get("IsListedCom")
        labels = split_multilabel(record.get("PunishmentType"))

        input_text = _join_sections(
            [
                ("违规行为", activity),
                ("法律依据", law),
                ("当事人", party),
                ("职位", position),
                ("与上市公司关系", relationship),
                ("发布机构", promulgator),
                ("公告日期", declare_date),
                ("是否上市公司", _normalize_binary_flag(is_listed)),
            ]
        )

        samples.append(
            {
                "sample_id": f"{event_id}:{number}",
                "event_id": event_id,
                "party_number": number,
                "party": party,
                "position": position,
                "relationship": relationship,
                "declare_date": declare_date,
                "promulgator": promulgator,
                "input_text": input_text,
                "labels": labels,
                "sum_penalty": record.get("SumPenalty"),
            }
        )
    return samples


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

from __future__ import annotations

from collections import Counter

from csrc_rag.data.excel_loader import iter_records, read_schema
from csrc_rag.data.schema import normalize_text
from csrc_rag.settings import RAW_WORKBOOK


def main() -> None:
    schema = read_schema(RAW_WORKBOOK)
    records = iter_records(RAW_WORKBOOK)

    event_ids = [str(record["EventID"]) for record in records if record.get("EventID") not in (None, "")]
    years = [normalize_text(record.get("DeclareDate"))[:4] for record in records if normalize_text(record.get("DeclareDate"))]
    punishments = [normalize_text(record.get("PunishmentType")) for record in records if normalize_text(record.get("PunishmentType"))]

    print("字段数:", len(schema.english_headers))
    print("数据行数:", len(records))
    print("唯一事件数:", len(set(event_ids)))
    print("年份范围:", min(years), max(years))
    print("字段名:", schema.english_headers)
    print("处罚方式 Top 10:")
    for label, count in Counter(punishments).most_common(10):
        print(" ", label, count)


if __name__ == "__main__":
    main()


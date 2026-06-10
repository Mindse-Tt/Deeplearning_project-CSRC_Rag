from __future__ import annotations

from dataclasses import dataclass


HEADER_ROW = 1
ZH_HEADER_ROW = 2
DATA_START_ROW = 4


@dataclass(frozen=True)
class WorkbookSchema:
    english_headers: list[str]
    chinese_headers: list[str]


def normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", "\n").strip()
    if not text:
        return None
    lines = [line.strip() for line in text.split("\n")]
    compact = "\n".join(line for line in lines if line)
    return compact or None


def split_multilabel(value: object) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []
    labels = [item.strip() for item in text.replace("，", ",").split(",")]
    return [item for item in labels if item]


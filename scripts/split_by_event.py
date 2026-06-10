"""按 EventID + 时间三分切分脚本（骨架）。

切分规则（与 docs/strategies/02-data-strategy.md 对齐）:
    - train: 1994 <= year <= 2021
    - val:   2022 <= year <= 2023
    - test:  2024 <= year <= 2025
    - unknown: 缺失日期 -> 默认进 train, 独立计数

主键:
    EventID。同一 EventID 的所有当事人 / chunk 必须落在同一 split。

时间基准:
    优先 SupervisionDate, 缺失回退到 DeclareDate, 再缺失 -> unknown。

输出:
    data/processed/splits/
        train.event_ids.txt
        val.event_ids.txt
        test.event_ids.txt
        unknown.event_ids.txt
        summary.json   # 每个桶的事件数 / 当事人数 / chunk 数

Usage:
    python scripts/split_by_event.py \
        --event-corpus data/processed/event_corpus.jsonl \
        --party-samples data/processed/party_samples.jsonl \
        --event-chunks data/processed/event_chunks.jsonl \
        --output-dir data/processed/splits
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

TRAIN_YEARS = range(1994, 2022)   # [1994, 2021]
VAL_YEARS = range(2022, 2024)     # [2022, 2023]
TEST_YEARS = range(2024, 2026)    # [2024, 2025]


@dataclass(frozen=True)
class SplitConfig:
    """切分配置（不可变）。"""

    event_corpus: Path
    party_samples: Path
    event_chunks: Path
    output_dir: Path


@dataclass(frozen=True)
class SplitBucket:
    """单个 split 桶的计数结果。"""

    name: str
    event_ids: frozenset[str]
    party_count: int
    chunk_count: int


def parse_args() -> SplitConfig:
    """解析命令行参数并返回不可变配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-corpus", type=Path, required=True)
    parser.add_argument("--party-samples", type=Path, required=True)
    parser.add_argument("--event-chunks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return SplitConfig(
        event_corpus=args.event_corpus,
        party_samples=args.party_samples,
        event_chunks=args.event_chunks,
        output_dir=args.output_dir,
    )


def iter_jsonl(path: Path) -> Iterable[dict]:
    """惰性读取 JSONL。"""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_year(event: dict) -> int | None:
    """从 event record 中提取年份。

    优先 supervision_date, 缺失回退 declare_date。返回 None 表示无法解析。
    """
    # TODO: 实现日期解析 + 健壮的错误兜底
    raise NotImplementedError


def classify_event(event: dict) -> str:
    """把单个 event 映射到 {train, val, test, unknown}。"""
    # TODO: 基于 extract_year 实现
    raise NotImplementedError


def build_buckets(config: SplitConfig) -> dict[str, SplitBucket]:
    """扫描 event_corpus, 产出 4 个 split 桶。"""
    # TODO: 读 event_corpus, 调 classify_event, 聚合 party / chunk 计数
    raise NotImplementedError


def write_buckets(buckets: dict[str, SplitBucket], output_dir: Path) -> None:
    """把 event_ids 写到 txt, 把 summary 写到 json。"""
    # TODO: 每个 split 写一个 event_ids.txt + 一个 summary.json
    raise NotImplementedError


def assert_no_leakage(buckets: dict[str, SplitBucket]) -> None:
    """交集断言: 任意两个桶的 event_ids 不得重叠。"""
    names = list(buckets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = buckets[a].event_ids & buckets[b].event_ids
            if overlap:
                raise AssertionError(
                    f"Leakage: {a} ∩ {b} = {len(overlap)} event_ids, 例: {list(overlap)[:3]}"
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = parse_args()
    buckets = build_buckets(config)
    assert_no_leakage(buckets)
    write_buckets(buckets, config.output_dir)
    logger.info("split done -> %s", config.output_dir)


if __name__ == "__main__":
    main()

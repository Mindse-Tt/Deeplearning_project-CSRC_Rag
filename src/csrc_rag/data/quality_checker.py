"""数据质量检查器（骨架）。

负责在数据构建之后、进入模型之前,对 data/processed/*.jsonl 做一轮质量审计:

    1. Schema / 类型 / 必填字段
    2. 去重 (精确 + MinHash 近似)
    3. 异常值 / 截断 / 离群
    4. 反泄漏 (PunishmentMeasure 不得进入 retrieval_text / input_text;
              val / test EventID 不得出现在 rag_qa_train.context 里)
    5. 覆盖率 / 长度分布

产出:
    QualityReport(dataclass)  + data/processed/quality_report.json

用法:
    from csrc_rag.data.quality_checker import QualityChecker
    report = QualityChecker(config).run()
    report.raise_on_critical()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# 禁止出现在任何模型输入文本里的字段名 (会泄漏标签)
FORBIDDEN_INPUT_FIELDS: frozenset[str] = frozenset({"PunishmentMeasure", "处分措施"})

# 必填字段 (按数据集划分)
REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "event_corpus": frozenset({"event_id", "retrieval_text"}),
    "event_chunks": frozenset({"event_id", "chunk_index", "text"}),
    "party_samples": frozenset({"sample_id", "event_id", "input_text"}),
    "rag_qa_train": frozenset({"qa_id", "event_id", "question", "answer", "split"}),
    "intent_train": frozenset({"id", "text", "intent", "split"}),
    "eval_gold": frozenset({"eid", "question", "intent", "expected_event_ids"}),
}

SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"


@dataclass(frozen=True)
class Finding:
    """单条质量问题。"""

    dataset: str
    rule: str
    severity: str
    message: str
    sample_ids: tuple[str, ...] = ()


@dataclass
class QualityReport:
    """质量报告（可变容器, 汇总用）。"""

    findings: list[Finding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_CRITICAL]

    def raise_on_critical(self) -> None:
        crit = self.critical()
        if crit:
            raise AssertionError(
                f"Quality check failed with {len(crit)} critical findings: "
                f"{[f.rule for f in crit[:3]]}"
            )

    def to_dict(self) -> dict:
        return {
            "counts": self.counts,
            "findings": [f.__dict__ for f in self.findings],
        }


@dataclass(frozen=True)
class QualityConfig:
    """质量检查配置。"""

    processed_dir: Path
    splits_dir: Path
    report_path: Path
    minhash_threshold: float = 0.9


class QualityChecker:
    """数据质量检查器入口。

    每条 check_* 方法对应一条规则, 产出 0+ Finding。
    主调用链 run() 负责组合所有规则并写出报告。
    """

    def __init__(self, config: QualityConfig) -> None:
        self._config = config
        self._report = QualityReport()

    # ------- 工具 -------

    def _iter_jsonl(self, name: str) -> Iterable[dict]:
        path = self._config.processed_dir / f"{name}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _load_split(self, split: str) -> frozenset[str]:
        path = self._config.splits_dir / f"{split}.event_ids.txt"
        if not path.exists():
            return frozenset()
        return frozenset(path.read_text(encoding="utf-8").splitlines())

    # ------- 规则 -------

    def check_required_fields(self, dataset: str) -> None:
        """必填字段缺失检测。"""
        # TODO: iterate rows, detect missing fields, add findings
        raise NotImplementedError

    def check_no_punishment_measure_leak(self, dataset: str) -> None:
        """断言模型输入文本不含 PunishmentMeasure / 处分措施。"""
        # TODO: 扫描 retrieval_text / input_text / context 字段
        raise NotImplementedError

    def check_split_leakage(self) -> None:
        """val/test EventID 不得出现在 rag_qa_train 的 context 里。"""
        # TODO: 读 splits + rag_qa_train, 计算交集
        raise NotImplementedError

    def check_exact_duplicates(self, dataset: str, key_fields: tuple[str, ...]) -> None:
        """精确去重检测。"""
        # TODO: 用 (field1, field2, ...) 做 key, 统计重复
        raise NotImplementedError

    def check_near_duplicates(self, dataset: str, text_field: str) -> None:
        """MinHash-LSH 近似去重, 阈值 self._config.minhash_threshold。"""
        # TODO: 用 datasketch 或手写 MinHash
        raise NotImplementedError

    def check_length_distribution(self, dataset: str, text_field: str) -> None:
        """记录 text 长度 p50 / p95 / p99, 超长打标。"""
        # TODO
        raise NotImplementedError

    def check_date_ranges(self) -> None:
        """event_corpus 的日期应落在 1994-2025; 越界打 warning。"""
        # TODO
        raise NotImplementedError

    # ------- 主调用 -------

    def run(self) -> QualityReport:
        """运行所有规则, 写出 JSON 报告, 返回 QualityReport。"""
        # TODO: 按数据集维度调度上面的 check_*, 最后 dump 到 self._config.report_path
        raise NotImplementedError

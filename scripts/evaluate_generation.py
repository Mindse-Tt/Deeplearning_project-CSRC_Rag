"""生成级评估 · 5 组对照 (G0-G4)。

组合：
  G0 · Qwen2.5-1.5B 原模型 无 RAG
  G1 · Qwen2.5-1.5B 原模型 + RAG (Hybrid+Rerank Top5)
  G2 · Qwen2.5-1.5B 原模型 + RAG + 强证据约束 Prompt
  G3 · Qwen2.5-1.5B LoRA  + RAG + 强证据约束 Prompt
  G4 · API 大模型 (DeepSeek / Qwen-Max) + RAG + 强证据约束 Prompt

自动指标:
  - ROUGE-L (字符 / jieba 双口径)
  - BERTScore-F1 (bert-base-chinese)
  - 引证命中率 (citation_hit)
  - 法条正确率 (law_jaccard)

半自动指标:
  - 幻觉率 hallucination_rate (regex + 与证据集合比对)

成本指标:
  - prompt / completion token
  - 推理延迟
  - 峰值显存 / 内存

当前为 stub, 请按 TODO 依序补齐。对应策略: docs/strategies/09-evaluation-strategy.md
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

# ---------- 常量 ----------

GROUP_LABELS = ("G0", "G1", "G2", "G3", "G4")
DEFAULT_TOP_K = 5

# 引用 / 法条抽取 regex
EVENT_ID_RE = re.compile(r"EVT[-_]?\d{4}[-_]?\d{3,6}")
LAW_RE = re.compile(r"《[^《》]{2,30}》第[一二三四五六七八九十百零〇\d]+条")


# ---------- 数据结构 ----------


@dataclass
class GoldItem:
    gold_id: str
    intent: str
    question: str
    gold_answer: str
    gold_event_ids: list[str]
    gold_laws: list[str]
    gold_punishment_types: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    trap: bool = False


@dataclass
class GenerationSample:
    gold_id: str
    group: str
    question: str
    answer: str
    evidence_event_ids: list[str]
    evidence_laws: list[str]
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    peak_memory_mb: float | None = None


@dataclass
class SampleMetrics:
    gold_id: str
    group: str
    rouge_l_char: float
    rouge_l_jieba: float
    bertscore_f1: float
    citation_hit: float
    law_jaccard: float
    hallucination: bool
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float


# ---------- I/O ----------


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- 指标 ----------


def rouge_l(ref: str, hyp: str, tokenizer: Callable[[str], list[str]]) -> float:
    """字符级 / jieba 切分 ROUGE-L (LCS / 参考长度)。"""
    # TODO: 真正计算 LCS；当前 stub 返回 0.0，测试时填充。
    _ = (ref, hyp, tokenizer)
    return 0.0


def bertscore_f1(refs: list[str], hyps: list[str], model_name: str = "bert-base-chinese") -> list[float]:
    """调用 bert_score 库 (懒加载避免环境缺包报错)。"""
    # TODO: from bert_score import score; P, R, F1 = score(hyps, refs, lang="zh", model_type=model_name)
    _ = (refs, hyps, model_name)
    return [0.0] * len(refs)


def citation_hit(answer: str, evidence_ids: list[str]) -> float:
    cited = set(EVENT_ID_RE.findall(answer))
    if not cited:
        return 0.0
    return len(cited & set(evidence_ids)) / len(cited)


def law_jaccard(answer: str, gold_laws: list[str]) -> float:
    predicted = set(LAW_RE.findall(answer))
    gold = set(gold_laws)
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    return len(predicted & gold) / len(predicted | gold)


def detect_hallucination(answer: str, evidence_event_ids: list[str], evidence_laws: list[str]) -> bool:
    """出现证据集合之外的 EventID 或法条则判为幻觉。"""
    evidence_events = set(evidence_event_ids)
    evidence_law_set = set(evidence_laws)
    for eid in EVENT_ID_RE.findall(answer):
        if eid not in evidence_events:
            return True
    for law in LAW_RE.findall(answer):
        if law not in evidence_law_set:
            return True
    return False


# ---------- 生成 pipeline stub ----------


def generate_group(group: str, gold: GoldItem, top_k: int = DEFAULT_TOP_K) -> GenerationSample:
    """TODO: 真正调用对应组的 pipeline。

    - G0: model.generate(question) 无检索
    - G1: retriever.search(question, k) + 默认 prompt + 原模型
    - G2: 同 G1 但用强证据约束 prompt
    - G3: 同 G2 但模型替换为 LoRA 权重 (peft.load_adapter)
    - G4: 调用 DeepSeek / Qwen-Max API
    """
    assert group in GROUP_LABELS, f"unknown group: {group}"
    start = time.perf_counter()
    answer = "<stub generated answer>"
    elapsed = time.perf_counter() - start
    return GenerationSample(
        gold_id=gold.gold_id,
        group=group,
        question=gold.question,
        answer=answer,
        evidence_event_ids=gold.gold_event_ids,  # stub: 真正实现需传入 retriever 输出
        evidence_laws=gold.gold_laws,
        prompt_tokens=0,
        completion_tokens=0,
        latency_sec=elapsed,
        peak_memory_mb=None,
    )


# ---------- 主流程 ----------


def evaluate_sample(gold: GoldItem, sample: GenerationSample) -> SampleMetrics:
    rouge_char = rouge_l(gold.gold_answer, sample.answer, list)
    rouge_jieba = rouge_l(gold.gold_answer, sample.answer, list)  # TODO: jieba.lcut
    bert_f1 = bertscore_f1([gold.gold_answer], [sample.answer])[0]
    hit = citation_hit(sample.answer, sample.evidence_event_ids)
    law_acc = law_jaccard(sample.answer, gold.gold_laws)
    hallu = detect_hallucination(sample.answer, sample.evidence_event_ids, sample.evidence_laws)
    return SampleMetrics(
        gold_id=gold.gold_id,
        group=sample.group,
        rouge_l_char=rouge_char,
        rouge_l_jieba=rouge_jieba,
        bertscore_f1=bert_f1,
        citation_hit=hit,
        law_jaccard=law_acc,
        hallucination=hallu,
        prompt_tokens=sample.prompt_tokens,
        completion_tokens=sample.completion_tokens,
        latency_sec=sample.latency_sec,
    )


def aggregate(metrics_rows: list[SampleMetrics]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[SampleMetrics]] = {g: [] for g in GROUP_LABELS}
    for row in metrics_rows:
        buckets[row.group].append(row)

    def mean_or_zero(xs: list[float]) -> float:
        return round(mean(xs), 4) if xs else 0.0

    summary: dict[str, dict[str, float]] = {}
    for group, rows in buckets.items():
        if not rows:
            continue
        summary[group] = {
            "n": len(rows),
            "ROUGE-L_char": mean_or_zero([r.rouge_l_char for r in rows]),
            "ROUGE-L_jieba": mean_or_zero([r.rouge_l_jieba for r in rows]),
            "BERTScore-F1": mean_or_zero([r.bertscore_f1 for r in rows]),
            "CitationHit": mean_or_zero([r.citation_hit for r in rows]),
            "LawJaccard": mean_or_zero([r.law_jaccard for r in rows]),
            "HallucinationRate": mean_or_zero([1.0 if r.hallucination else 0.0 for r in rows]),
            "AvgPromptTokens": mean_or_zero([float(r.prompt_tokens) for r in rows]),
            "AvgCompletionTokens": mean_or_zero([float(r.completion_tokens) for r in rows]),
            "AvgLatencySec": mean_or_zero([r.latency_sec for r in rows]),
        }
    return summary


def run(gold_path: Path, output_dir: Path, groups: tuple[str, ...], top_k: int) -> None:
    gold_rows = load_jsonl(gold_path)
    gold_items = [GoldItem(**{k: row[k] for k in GoldItem.__dataclass_fields__ if k in row}) for row in gold_rows]

    all_samples: list[GenerationSample] = []
    all_metrics: list[SampleMetrics] = []
    for gold in gold_items:
        if gold.trap:
            continue
        for group in groups:
            sample = generate_group(group, gold, top_k=top_k)
            metrics = evaluate_sample(gold, sample)
            all_samples.append(sample)
            all_metrics.append(metrics)

    dump_jsonl(output_dir / "generation_samples.jsonl", [asdict(s) for s in all_samples])
    dump_jsonl(output_dir / "generation_metrics.jsonl", [asdict(m) for m in all_metrics])

    summary = aggregate(all_metrics)
    (output_dir / "generation_matrix.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="5 组对照生成评估")
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_50.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--groups", nargs="+", default=list(GROUP_LABELS))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.gold, args.out, tuple(args.groups), args.top_k)


if __name__ == "__main__":
    main()

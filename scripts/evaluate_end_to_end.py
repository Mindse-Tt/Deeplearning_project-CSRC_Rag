"""端到端人工评估 · 汇总 5 位标注员的 blind 打分。

输入:
  --gold   data/eval/gold_50.jsonl
  --scores data/eval/human_scores.jsonl   (5 位组员填写后合并)

输出:
  artifacts/eval/end_to_end_scores.json   每组 × 每维度均值 + κ
  artifacts/eval/end_to_end_per_item.jsonl 每条金标的聚合

维度:
  correctness 1-5 / completeness 1-5 / fluency 1-5 / hallucination Y/N

一致性:
  对每个维度计算两两 Cohen's κ (或 squared-weighted κ for 1-5), 再取均值作为组的一致性指标
  另外计算 Fleiss' κ 作为辅助

当前为 stub, 请按 TODO 补齐。对应策略: docs/strategies/09-evaluation-strategy.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


SCORE_DIMENSIONS = ("correctness", "completeness", "fluency")
BOOL_DIMENSIONS = ("hallucination",)
MIN_ANNOTATORS_PER_ITEM = 3


@dataclass
class HumanScore:
    gold_id: str
    model_label: str
    true_model: str
    annotator: str
    correctness: int
    completeness: int
    fluency: int
    hallucination: bool
    comment: str = ""


# ---------- I/O ----------


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- κ 计算 ----------


def cohens_kappa(pairs: list[tuple[Any, Any]], categories: list[Any]) -> float:
    """标准 Cohen's κ (1960). pairs: [(rater_a, rater_b), ...]"""
    if not pairs:
        return float("nan")
    n = len(pairs)
    obs_agree = sum(1 for a, b in pairs if a == b) / n
    a_dist = {c: 0 for c in categories}
    b_dist = {c: 0 for c in categories}
    for a, b in pairs:
        if a in a_dist:
            a_dist[a] += 1
        if b in b_dist:
            b_dist[b] += 1
    exp_agree = sum((a_dist[c] / n) * (b_dist[c] / n) for c in categories)
    if exp_agree == 1.0:
        return 1.0
    return (obs_agree - exp_agree) / (1.0 - exp_agree)


def average_pairwise_kappa(
    per_item_scores: dict[str, dict[str, Any]],
    dimension: str,
    categories: list[Any],
) -> float:
    """对每条 item, 取所有 annotator 两两配对, 汇总到全局 pair-list, 再计算 κ。"""
    all_pairs: list[tuple[Any, Any]] = []
    for _gold_id, anno_map in per_item_scores.items():
        annos = list(anno_map.values())
        for a, b in combinations(annos, 2):
            if dimension in a and dimension in b:
                all_pairs.append((a[dimension], b[dimension]))
    return cohens_kappa(all_pairs, categories)


# ---------- 聚合 ----------


def aggregate_per_item(
    scores: list[HumanScore],
) -> dict[tuple[str, str], dict[str, Any]]:
    """按 (gold_id, true_model) 聚合, 每条取均值 + 多数票幻觉判定。"""
    bucket: dict[tuple[str, str], list[HumanScore]] = defaultdict(list)
    for s in scores:
        bucket[(s.gold_id, s.true_model)].append(s)

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in bucket.items():
        if len(rows) < MIN_ANNOTATORS_PER_ITEM:
            # 仍然记录, 但加 low_coverage 标记
            low_cov = True
        else:
            low_cov = False
        result[key] = {
            "n_annotators": len(rows),
            "low_coverage": low_cov,
            "correctness": mean(r.correctness for r in rows),
            "completeness": mean(r.completeness for r in rows),
            "fluency": mean(r.fluency for r in rows),
            "hallucination_rate": sum(1 for r in rows if r.hallucination) / len(rows),
            "hallucination_majority": sum(1 for r in rows if r.hallucination) > len(rows) / 2,
        }
    return result


def aggregate_per_group(
    per_item: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, dict[str, float]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_gold_id, model), row in per_item.items():
        by_group[model].append(row)

    summary: dict[str, dict[str, float]] = {}
    for model, rows in by_group.items():
        summary[model] = {
            "n_items": len(rows),
            "correctness_mean": round(mean(r["correctness"] for r in rows), 3),
            "completeness_mean": round(mean(r["completeness"] for r in rows), 3),
            "fluency_mean": round(mean(r["fluency"] for r in rows), 3),
            "hallucination_rate_mean": round(mean(r["hallucination_rate"] for r in rows), 3),
            "hallucination_majority_rate": round(
                sum(1 for r in rows if r["hallucination_majority"]) / len(rows), 3
            ),
        }
    return summary


# ---------- κ 汇总 ----------


def compute_agreement(scores: list[HumanScore]) -> dict[str, float]:
    # 构造 per_item_scores: {gold_id: {annotator: {dim: value}}}
    per_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for s in scores:
        key = f"{s.gold_id}|{s.true_model}"
        per_item[key][s.annotator] = {
            "correctness": s.correctness,
            "completeness": s.completeness,
            "fluency": s.fluency,
            "hallucination": bool(s.hallucination),
        }
    report: dict[str, float] = {}
    for dim in SCORE_DIMENSIONS:
        report[f"kappa_{dim}"] = round(
            average_pairwise_kappa(per_item, dim, categories=[1, 2, 3, 4, 5]), 3
        )
    for dim in BOOL_DIMENSIONS:
        report[f"kappa_{dim}"] = round(
            average_pairwise_kappa(per_item, dim, categories=[True, False]), 3
        )
    return report


# ---------- 主流程 ----------


def run(gold_path: Path, scores_path: Path, output_dir: Path) -> None:
    gold_rows = load_jsonl(gold_path)
    gold_index = {row["gold_id"]: row for row in gold_rows}

    raw_scores = load_jsonl(scores_path)
    scores = [
        HumanScore(
            gold_id=row["gold_id"],
            model_label=row["model_label"],
            true_model=row["true_model"],
            annotator=row["annotator"],
            correctness=int(row["correctness"]),
            completeness=int(row["completeness"]),
            fluency=int(row["fluency"]),
            hallucination=bool(row["hallucination"]),
            comment=row.get("comment", ""),
        )
        for row in raw_scores
        if row["gold_id"] in gold_index  # 过滤陷阱样本外无效行
    ]

    per_item = aggregate_per_item(scores)
    per_group = aggregate_per_group(per_item)
    agreement = compute_agreement(scores)

    dump_jsonl(
        output_dir / "end_to_end_per_item.jsonl",
        [
            {"gold_id": gid, "true_model": model, **info}
            for (gid, model), info in per_item.items()
        ],
    )
    (output_dir / "end_to_end_scores.json").write_text(
        json.dumps(
            {
                "per_group": per_group,
                "agreement": agreement,
                "n_total_scores": len(scores),
                "n_items": len(gold_index),
                "notes": "κ < 0.6 时需重新校准标注者口径, 参见 09-evaluation-strategy.md",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"per_group": per_group, "agreement": agreement}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="端到端人工评估汇总")
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_50.jsonl"))
    parser.add_argument("--scores", type=Path, default=Path("data/eval/human_scores.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.gold, args.scores, args.out)


if __name__ == "__main__":
    main()

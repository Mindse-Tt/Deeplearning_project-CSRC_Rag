"""补充实验图表生成脚本（团队人工编写）。

把仓库里已有数字、但此前缺少成图的几组结果补成论文级图（输出 docs/visuals/png/experiments/）：

  1. finetune_6metrics.png        —— 最终 6 指标分组柱状图：Qwen2.5-0.5B+LoRA vs Bloom-560M+QLoRA
  2. finetune_6metrics_radar.png  —— 同上的雷达图（幻觉率取「非幻觉率」以统一"越大越好"）
  3. train_category_dist.png      —— 训练集 A–H 八类样本配比（柱状图）
  4. latency_g0_g3.png            —— G0–G3 端到端平均延迟对比

数据来源（均为仓库内已落盘的真实数据 / 定稿口径）：
  - 6 指标：README §4.3 最终总表（n=50），系统三项与 docs/reports/m4_4_generation_eval.json 的 G3 一致。
  - A–H 配比：data/processed/rag_qa_train.jsonl 按 category 实时统计（合计 5,360）。
  - 延迟：docs/reports/m4_4_generation_eval.json 的 G0/G1/G2/G3 avg_latency_s。

用法：
    PYTHONPATH=src python scripts/build_extra_figures.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，便于在服务器/CI 直接出图
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "visuals" / "png" / "experiments"
TRAIN_JSONL = ROOT / "data" / "processed" / "rag_qa_train.jsonl"
GEN_EVAL = ROOT / "docs" / "reports" / "m4_4_generation_eval.json"

# 中文字体：优先 Windows(YaHei)，其次 macOS 常见黑体，最后兜底，保证跨平台都能渲染中文
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei",
            "PingFang SC",
            "Heiti TC",
            "STHeiti",
            "Arial Unicode MS",
            "Hiragino Sans GB",
            "Arial",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "font.size": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

C_BLUE = "#2b6cb0"
C_GREY = "#9aa0a6"

# ---------------------------------------------------------------------------
# 图 1/2：最终 6 项指标（README §4.3 定稿口径，n=50）
# ---------------------------------------------------------------------------
METRIC_LABELS = ["任务准确率↑", "实体F1↑", "指令遵循↑", "格式合规↑", "EID命中↑", "幻觉率↓"]
SIX_METRICS = {
    "Qwen2.5-0.5B + LoRA": [0.28, 0.52, 0.76, 0.76, 0.28, 0.02],
    "Bloom-560M + QLoRA": [0.00, 0.04, 0.08, 0.08, 0.00, 0.793],
}
SIX_COLORS = {"Qwen2.5-0.5B + LoRA": C_BLUE, "Bloom-560M + QLoRA": C_GREY}


def fig_grouped_bars() -> None:
    x = np.arange(len(METRIC_LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, name in enumerate(SIX_METRICS):
        bars = ax.bar(x + (i - 0.5) * width, SIX_METRICS[name], width,
                      label=name, color=SIX_COLORS[name], edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, SIX_METRICS[name]):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=9.5)
    ax.set_ylabel("指标值")
    ax.set_ylim(0, 1.0)
    ax.set_title("最终 6 项指标对比（Qwen2.5-0.5B+LoRA vs Bloom-560M+QLoRA，n=50）", pad=8)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "finetune_6metrics.png")
    plt.close(fig)
    print("已生成: docs/visuals/png/experiments/finetune_6metrics.png")


def fig_radar() -> None:
    labels = METRIC_LABELS.copy()
    labels[-1] = "非幻觉率↑"

    def to_radar(vals: list[float]) -> list[float]:
        v = vals.copy()
        v[-1] = 1.0 - v[-1]
        return v

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.4, 6.0), subplot_kw={"polar": True})
    for name in SIX_METRICS:
        vals = to_radar(SIX_METRICS[name])
        vals += vals[:1]
        ax.plot(angles, vals, label=name, color=SIX_COLORS[name], linewidth=2)
        ax.fill(angles, vals, color=SIX_COLORS[name], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5)
    ax.set_title("最终 6 项指标雷达图（越外越好；幻觉率已转为非幻觉率）", pad=18)
    ax.legend(loc="lower right", bbox_to_anchor=(1.18, -0.05), frameon=False)
    fig.savefig(OUT / "finetune_6metrics_radar.png")
    plt.close(fig)
    print("已生成: docs/visuals/png/experiments/finetune_6metrics_radar.png")


# ---------------------------------------------------------------------------
# 图 3：训练集 A–H 八类配比（从 rag_qa_train.jsonl 实时统计）
# ---------------------------------------------------------------------------
CATEGORY_NAMES = {
    "A": "A 案例检索",
    "B": "B 法条依据",
    "C": "C 处罚推荐",
    "D": "D 趋势分析",
    "E": "E 越界拒答",
    "F": "F 证据不足",
    "G": "G 多轮对话",
    "H": "H 反幻觉负样本",
}


def fig_category_dist() -> None:
    counts = collections.Counter()
    with TRAIN_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                counts[json.loads(line).get("category", "?")] += 1
    keys = [k for k in CATEGORY_NAMES if k in counts]
    labels = [CATEGORY_NAMES[k] for k in keys]
    values = [counts[k] for k in keys]
    total = sum(counts.values())

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    # E/F/H 三类反幻觉/拒答负样本用浅蓝高亮，体现"会拒答、少幻觉"的关键数据设计
    colors = [(C_BLUE if k in ("E", "F", "H") else C_GREY) for k in keys]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, values):
        ax.annotate(f"{v}\n({v / total:.0%})", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 1), textcoords="offset points")
    ax.set_ylabel("训练样本数")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title(f"训练集 A–H 八类样本配比（合计 {total:,} 条；蓝色为拒答/反幻觉负样本）", pad=8)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "train_category_dist.png")
    plt.close(fig)
    print(f"已生成: docs/visuals/png/experiments/train_category_dist.png （合计 {total} 条）")


# ---------------------------------------------------------------------------
# 图 4：G0–G3 端到端平均延迟（从 m4_4_generation_eval.json 读取）
# ---------------------------------------------------------------------------
def fig_latency() -> None:
    summary = json.loads(GEN_EVAL.read_text(encoding="utf-8"))["summary"]
    groups = ["G0", "G1", "G2", "G3"]
    desc = {"G0": "裸模型", "G1": "+RAG", "G2": "+强prompt", "G3": "+LoRA"}
    lat = [summary[g]["avg_latency_s"] for g in groups]
    labels = [f"{g}\n{desc[g]}" for g in groups]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = [C_GREY, C_GREY, C_GREY, C_BLUE]
    bars = ax.bar(labels, lat, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    for b, v in zip(bars, lat):
        ax.annotate(f"{v:.2f}s", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 1), textcoords="offset points")
    ax.set_ylabel("端到端平均延迟 (秒)")
    ax.set_ylim(0, max(lat) * 1.18)
    ax.set_title("G0–G3 端到端平均延迟对比（n=50）", pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "latency_g0_g3.png")
    plt.close(fig)
    print("已生成: docs/visuals/png/experiments/latency_g0_g3.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_grouped_bars()
    fig_radar()
    fig_category_dist()
    fig_latency()


if __name__ == "__main__":
    main()

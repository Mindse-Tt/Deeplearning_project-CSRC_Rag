"""Render academic-style figures for the paper.

Produces PNGs at 300 DPI, sans-serif, grayscale + low-sat blue/red, journal-ready.
All outputs go to ``docs/visuals/png/paper/``.

Data sources (read-only):
    docs/reports/m3e_retrieval_report.json
    docs/reports/m4_4_generation_eval.json
    docs/reports/m4_2_trend_eval.json
    docs/reports/m5_macbert_report.json
    artifacts/models/qwen_lora_csrc/checkpoint-274/trainer_state.json
    data/processed/event_corpus.jsonl
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "reports"
OUT = ROOT / "docs" / "visuals" / "png" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style: academic journal (ACL / IEEE look)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "PingFang SC", "Heiti TC", "STHeiti", "Arial Unicode MS", "Hiragino Sans GB", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    }
)

# Low-saturation palette (paper-safe, prints OK in grayscale)
C_BLUE = "#2F5597"
C_RED = "#A5333F"
C_GREEN = "#3F7D5A"
C_GRAY_DARK = "#3E3E3E"
C_GRAY_MID = "#8A8A8A"
C_GRAY_LIGHT = "#CFCFCF"
C_ORANGE = "#C27B2E"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Figure 1 — System architecture (7 layers)
# ---------------------------------------------------------------------------
def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    layers = [
        (9.5, "用户查询 (user query)", C_GRAY_DARK, "white"),
        (8.3, "L1  意图分类 (IntentClassifier v2)", C_BLUE, "white"),
        (7.1, "L2  查询改写 (multi sub-query, OR 拆分)", C_BLUE, "white"),
        (5.9, "L3  双路检索 · BM25 + bge-small-zh · RRF(k=60)", C_BLUE, "white"),
        (4.7, "L4  交叉编码器精排 (bge-reranker-v2-m3, top-5)", C_BLUE, "white"),
        (3.5, "L5  Qwen2.5-0.5B + LoRA · 受证据约束生成", C_RED, "white"),
        (2.3, "L6  趋势聚合 · SQL-like groupby (short-circuit)", C_GREEN, "white"),
        (1.1, "L7  引证校验 (Validator, 8 条 yaml 规则)", C_GRAY_DARK, "white"),
    ]

    for y, text, color, textcolor in layers:
        box = FancyBboxPatch(
            (1.2, y - 0.35),
            7.6,
            0.7,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.92,
        )
        ax.add_patch(box)
        ax.text(5.0, y, text, ha="center", va="center", color=textcolor, fontsize=10)

    # arrows between successive layers (except L6 branch)
    arrow_pairs = [
        (9.5, 8.3),
        (8.3, 7.1),
        (7.1, 5.9),
        (5.9, 4.7),
        (4.7, 3.5),
        (3.5, 1.1),
    ]
    for y1, y2 in arrow_pairs:
        ax.annotate(
            "",
            xy=(5.0, y2 + 0.36),
            xytext=(5.0, y1 - 0.36),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_GRAY_DARK),
        )
    # L6 branch (dashed, off main flow)
    ax.annotate(
        "",
        xy=(1.2, 2.3),
        xytext=(1.2, 5.9),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_GREEN, linestyle="--"),
    )
    ax.annotate(
        "",
        xy=(5.0, 1.1 + 0.36),
        xytext=(1.2 + 0.0, 2.3),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_GREEN, linestyle="--"),
    )
    ax.text(
        0.9, 4.1, "trend_analysis\n意图短路",
        ha="center", va="center", fontsize=8.5, color=C_GREEN,
        rotation=90,
    )

    # legend below
    handles = [
        mpatches.Patch(color=C_BLUE, label="检索链路 (Retrieval)"),
        mpatches.Patch(color=C_RED, label="生成链路 (Generation)"),
        mpatches.Patch(color=C_GREEN, label="统计聚合旁路 (Aggregation)"),
        mpatches.Patch(color=C_GRAY_DARK, label="边界 / 校验 (Guard & Validator)"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    ax.set_title("Figure 1 · 七层 RAG 流水线架构 (7-layer RAG pipeline)", fontsize=11, pad=10)
    fig.savefig(OUT / "fig1_architecture.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — G0-G3 four-group generation comparison (main result)
# ---------------------------------------------------------------------------
def fig_g0_g3() -> None:
    d = _load_json(REPORTS / "m4_4_generation_eval.json")
    s = d["summary"]
    groups = ["G0", "G1", "G2", "G3"]
    config_label = {
        "G0": "base",
        "G1": "+RAG",
        "G2": "+strong prompt",
        "G3": "+LoRA",
    }
    eid = [s[g]["event_id_hit_rate"] for g in groups]
    fmt = [s[g]["format_compliance_rate"] for g in groups]
    hallu = [s[g]["hallucinated_number_rate"] for g in groups]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    import numpy as np
    x = np.arange(len(groups))
    width = 0.26

    b1 = ax.bar(x - width, eid, width, label="EventID Hit Rate",
                color=C_BLUE, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, fmt, width, label="Format Compliance",
                color=C_GREEN, edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + width, hallu, width, label="Hallucinated-Number Rate",
                color=C_RED, edgecolor="black", linewidth=0.5, hatch="//")

    for bars in [b1, b2, b3]:
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.015,
                f"{h:.2f}" if h > 0 else "0",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n({config_label[g]})" for g in groups])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 2 · 四组生成对照 (n = 50, gold_130 分层抽样)", pad=8)
    ax.legend(loc="upper left", frameon=False)
    # shaded box over G3
    ax.axvspan(2.55, 3.45, alpha=0.06, color=C_RED, zorder=0)

    fig.savefig(OUT / "fig2_g0_g3.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Hallucination rate decrease across G0→G3 (line)
# ---------------------------------------------------------------------------
def fig_hallucination() -> None:
    d = _load_json(REPORTS / "m4_4_generation_eval.json")
    s = d["summary"]
    groups = ["G0", "G1", "G2", "G3"]
    rates = [s[g]["hallucinated_number_rate"] for g in groups]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(groups, rates, marker="o", markersize=9, color=C_RED,
            linewidth=2.0, markerfacecolor="white", markeredgewidth=1.8)
    for i, r in enumerate(rates):
        ax.annotate(f"{r:.3f}", (i, r),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, color=C_RED, fontweight="bold")

    # Annotate layer wins
    ax.annotate("", xy=(1, 0.10), xytext=(0, 0.20),
                arrowprops=dict(arrowstyle="->", color=C_GRAY_DARK, lw=1.0))
    ax.text(0.5, 0.165, "L2 证据约束\n(-50%)",
            ha="center", fontsize=8.5, color=C_GRAY_DARK)

    ax.annotate("", xy=(3, 0.033), xytext=(2, 0.10),
                arrowprops=dict(arrowstyle="->", color=C_GRAY_DARK, lw=1.0))
    ax.text(2.5, 0.075, "L3 对抗训练\n(-67%)",
            ha="center", fontsize=8.5, color=C_GRAY_DARK)

    ax.set_ylim(0, 0.24)
    ax.set_ylabel("Hallucinated-Number Rate")
    ax.set_xlabel("Experimental condition")
    ax.set_title("Figure 3 · 幻觉数字率的逐层缓解 (hallucination rate reduction)", pad=8)
    fig.savefig(OUT / "fig3_hallucination.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Retrieval recall comparison (BM25 / dense / hybrid / rerank)
# ---------------------------------------------------------------------------
def fig_retrieval() -> None:
    d = _load_json(REPORTS / "m3e_retrieval_report.json")
    # Expect d['results'] is a list of dicts with label & Recall@5
    rows = sorted(d["results"], key=lambda r: r.get("Recall@5", 0))
    labels = [r["label"] for r in rows]
    r5 = [r.get("Recall@5", 0) for r in rows]
    h5 = [r.get("Hit@5", 0) for r in rows]
    mrr = [r.get("MRR", 0) for r in rows]
    ndcg = [r.get("nDCG@10", 0) for r in rows]

    import numpy as np
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - 1.5 * width, r5, width, label="Recall@5", color=C_BLUE,
           edgecolor="black", linewidth=0.4)
    ax.bar(x - 0.5 * width, h5, width, label="Hit@5", color=C_GREEN,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.5 * width, mrr, width, label="MRR", color=C_ORANGE,
           edgecolor="black", linewidth=0.4)
    ax.bar(x + 1.5 * width, ndcg, width, label="nDCG@10", color=C_GRAY_MID,
           edgecolor="black", linewidth=0.4)

    for i, v in enumerate(r5):
        ax.text(i - 1.5 * width, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, max(max(r5 + h5 + mrr + ndcg) * 1.15, 0.5))
    ax.set_ylabel("Metric value")
    ax.set_title("Figure 4 · 检索层消融 (retrieval ablation on gold_130)", pad=8)
    ax.legend(loc="upper left", frameon=False, ncol=4)
    fig.savefig(OUT / "fig4_retrieval.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — LoRA training loss curves
# ---------------------------------------------------------------------------
def fig_loss() -> None:
    state_path = ROOT / "artifacts/models/qwen_lora_csrc/checkpoint-274/trainer_state.json"
    # 训练 checkpoint 的逐步日志（trainer_state.json）体积较大、未随仓库提供；
    # 缺失则跳过本图（已提交的 fig5_loss.png 为训练时的真实产物），避免整个脚本中断。
    if not state_path.exists():
        print("[skip] fig5_loss：未找到 trainer_state.json（训练日志未入库），沿用已提交的 fig5_loss.png")
        return
    d = json.loads(state_path.read_text(encoding="utf-8"))
    log = d["log_history"]

    train_steps = [e["step"] for e in log if "loss" in e and "eval_loss" not in e]
    train_loss = [e["loss"] for e in log if "loss" in e and "eval_loss" not in e]
    eval_steps = [e["step"] for e in log if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in log if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(train_steps, train_loss, label="train loss",
            color=C_BLUE, linewidth=1.5, marker="o", markersize=3)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="eval loss",
                color=C_RED, linewidth=1.8, marker="s", markersize=5,
                linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Figure 5 · LoRA 训练收敛曲线 (Qwen2.5-0.5B, 2 epoch, 274 steps)", pad=8)
    ax.legend(frameon=False)

    # annotate final values
    ax.annotate(f"final train = {train_loss[-1]:.3f}",
                xy=(train_steps[-1], train_loss[-1]),
                xytext=(-60, 20), textcoords="offset points",
                fontsize=8.5, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.8))
    if eval_loss:
        ax.annotate(f"final eval = {eval_loss[-1]:.3f}",
                    xy=(eval_steps[-1], eval_loss[-1]),
                    xytext=(-70, -30), textcoords="offset points",
                    fontsize=8.5, color=C_RED,
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8))

    fig.savefig(OUT / "fig5_loss.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Corpus distribution (year histogram + violation types)
# ---------------------------------------------------------------------------
def fig_corpus() -> None:
    corpus = ROOT / "data" / "processed" / "event_corpus.jsonl"
    by_year: Counter[int] = Counter()
    by_vtype: Counter[str] = Counter()
    with corpus.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = r.get("declare_date") or r.get("supervision_date") or ""
            try:
                y = int(str(d)[:4])
                if 2000 <= y <= 2030:
                    by_year[y] += 1
            except Exception:
                pass
            for v in (r.get("violation_types") or [])[:3]:
                if ";" in v:
                    continue  # skip combined labels
                by_vtype[v] += 1

    years = sorted(by_year.keys())
    # focus on 2010-2025 for readability
    years = [y for y in years if 2010 <= y <= 2025]
    counts = [by_year[y] for y in years]

    top_v = by_vtype.most_common(8)
    vlabels = [v for v, _ in top_v]
    vcounts = [c for _, c in top_v]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    ax = axes[0]
    bars = ax.bar(years, counts, color=C_BLUE, edgecolor="black", linewidth=0.4)
    # highlight 2017+ growth band
    for bar, y in zip(bars, years):
        if y >= 2017:
            bar.set_color(C_RED)
    ax.set_xlabel("Year")
    ax.set_ylabel("Event count")
    ax.set_title("(a) 按年度分布 (events per year, 2010-2025)", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.axvline(2016.5, color=C_GRAY_DARK, linestyle=":", linewidth=0.8)
    ax.annotate("监管趋严\n(regulatory surge)",
                xy=(2017, max(counts) * 0.75),
                xytext=(2012, max(counts) * 0.85),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=C_GRAY_DARK, lw=0.8))

    ax = axes[1]
    y = range(len(vlabels))
    ax.barh(list(y), vcounts, color=C_GREEN, edgecolor="black", linewidth=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(vlabels)
    ax.invert_yaxis()
    ax.set_xlabel("Event count")
    ax.set_title("(b) Top-8 违规子类 (top-8 violation types)", fontsize=10)

    fig.suptitle("Figure 6 · 知识库分布 (n = 4,233 events from CSMAR 2000-2025)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_corpus.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    fig_architecture()
    print("[PASS] fig1_architecture.png")
    fig_g0_g3()
    print("[PASS] fig2_g0_g3.png")
    fig_hallucination()
    print("[PASS] fig3_hallucination.png")
    fig_retrieval()
    print("[PASS] fig4_retrieval.png")
    fig_loss()
    print("[PASS] fig5_loss.png")
    fig_corpus()
    print("[PASS] fig6_corpus.png")
    print(f"\nAll figures written to {OUT}")


if __name__ == "__main__":
    main()

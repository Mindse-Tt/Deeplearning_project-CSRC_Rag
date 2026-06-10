"""L3 混合检索融合：倒数排名融合（Reciprocal Rank Fusion, RRF）。

本模块是我们 BM25 词面通道与稠密向量通道的"会合点"。两条通道的原始分数量纲
完全不同（BM25 是无界正实数，余弦相似度落在 [0,1]），直接加权相加会被量纲主导。
RRF 只取各通道内的"名次"而非原始分，天然消除量纲差异、对异常分值鲁棒，是本项目
选用它做多路融合的核心理由。同一套 RRF 也被复用到 engine 里的子查询融合与
hybrid⊕rerank 事件级融合，保持全流水线融合语义一致。
"""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    # 对每个候选累加 1/(rrf_k + rank) 的贡献：rank 越靠前贡献越大。
    # rrf_k=60 是 RRF 原论文（Cormack 2009）的推荐常数，作用是"压平"头部
    # 名次之间的差距，避免某一路通道把它的 Top-1 强行钉死在融合榜首，
    # 从而让两路通道都召回的文档（双重命中）更容易冒头。本项目沿用该经验值。
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (rrf_k + rank)
    # 注意：此处刻意忽略各通道的原始 _score，只用名次，这正是 RRF 抗量纲的关键。
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


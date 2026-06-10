from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(ranked_event_ids: list[str], gold_event_id: str, k: int) -> float:
    return 1.0 if gold_event_id in ranked_event_ids[:k] else 0.0


def reciprocal_rank(ranked_event_ids: list[str], gold_event_id: str) -> float:
    for idx, event_id in enumerate(ranked_event_ids, start=1):
        if event_id == gold_event_id:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked_event_ids: list[str], gold_event_id: str, k: int) -> float:
    for idx, event_id in enumerate(ranked_event_ids[:k], start=1):
        if event_id == gold_event_id:
            return 1.0 / (idx.bit_length())
    return 0.0


# ---------------------------------------------------------------------------
# Multi-gold metrics (M3 eval set has ``relevant_event_ids`` which is a set
# of acceptable answers per question, not a single gold id).
# ---------------------------------------------------------------------------


def recall_at_k_multi(
    ranked_event_ids: list[str], gold_event_ids: Iterable[str], k: int
) -> float:
    gold = set(gold_event_ids)
    if not gold:
        return 0.0
    top = ranked_event_ids[:k]
    hits = sum(1 for e in top if e in gold)
    return hits / len(gold)


def hit_at_k_multi(
    ranked_event_ids: list[str], gold_event_ids: Iterable[str], k: int
) -> float:
    """Binary: 1 if any gold id appears in the top-k."""
    gold = set(gold_event_ids)
    if not gold:
        return 0.0
    return 1.0 if any(e in gold for e in ranked_event_ids[:k]) else 0.0


def reciprocal_rank_multi(
    ranked_event_ids: list[str], gold_event_ids: Iterable[str]
) -> float:
    gold = set(gold_event_ids)
    if not gold:
        return 0.0
    for idx, event_id in enumerate(ranked_event_ids, start=1):
        if event_id in gold:
            return 1.0 / idx
    return 0.0


def ndcg_at_k_multi(
    ranked_event_ids: list[str], gold_event_ids: Iterable[str], k: int
) -> float:
    """Binary-gain nDCG@k over a set of relevant ids.

    gains_i = 1 if ranked[i] in gold else 0
    DCG = sum(gain_i / log2(i+1))
    IDCG = sum_{i=1..min(|gold|,k)} 1 / log2(i+1)
    """
    gold = set(gold_event_ids)
    if not gold:
        return 0.0
    dcg = 0.0
    for idx, event_id in enumerate(ranked_event_ids[:k], start=1):
        if event_id in gold:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg



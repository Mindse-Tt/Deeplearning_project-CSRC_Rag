"""拒答策略层 —— 面向越界/低置信 query 的多级兜底决策。

本模块是我们为保证回答可信度而设计的"多级安全闸门"，在七层流水线的四个关键节点
分别拦截并降级，统一产出 ``IntentDecision`` 决策对象（设计见
``docs/strategies/01-reject-strategy.md``）：

  * L1 入口拒答：融合 ``intent_model``（ML 分类置信度）与 ``topic_guard``
    （硬关键词规则）双信号，命中越界/闲聊/招呼则直接 reject，不进入检索；
  * L3 检索兜底：召回过稀（命中数/最高分不达标）时降级，提示用户细化问题；
  * L5 生成兜底：生成置信度（mean logprob）过低或含模棱两可措辞时，
    退化为"只列最相关案例标题"而非给出不可靠结论；
  * L7 引用兜底：答案的证据可追溯率不达标时，只输出可核验的原文片段、不做推断。

设计约束：本文件只消费上述模块的公开 API，刻意不修改
``intents.py`` / ``intent_model.py`` / ``topic_guard.py``，保持单向依赖、职责清晰。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

IntentLabel = Literal[
    "greeting",
    "chitchat",
    "out_of_scope",
    "case_retrieval",
    "law_grounding",
    "sanction_recommendation",
    "trend_analysis",
]

ActionKind = Literal["reject", "retrieve", "aggregate"]
FallbackLevel = Literal["L1", "L3", "L5", "L7"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentDecision:
    """Unified decision object passed along the pipeline.

    Attributes:
        intent: One of the 7 intent labels.
        action: ``reject`` halts the pipeline; ``retrieve`` enters L2+L3;
            ``aggregate`` enters L6 trend-analysis path.
        confidence: Classifier confidence in [0, 1].
        fallback_message: User-facing Chinese message when action is reject.
        fallback_level: Which level triggered the fallback (L1/L3/L5/L7).
        debug: Free-form debug payload for offline analysis.
    """

    intent: IntentLabel
    action: ActionKind
    confidence: float
    fallback_message: str | None = None
    fallback_level: FallbackLevel | None = None
    debug: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fallback message templates (see docs/strategies/01-reject-strategy.md §5)
# ---------------------------------------------------------------------------

FALLBACK_MESSAGES: dict[str, str] = {
    "greeting": (
        "您好！我是证监会处罚案例问答助手，可以帮您查案例、查法条、看趋势。"
        "试试问我：『2023 年内幕交易典型案例』。"
    ),
    "chitchat": (
        "抱歉，我专注于证监会违规案例问答，无法进行闲聊或文学创作。"
        "您可以问我：某类违规的典型处罚、某年度的处罚趋势等。"
    ),
    "out_of_scope": (
        "该问题不在本系统支持范围内。本系统仅覆盖证监会证券违规处罚案例，"
        "保险、银行、生活类问题暂不支持。"
    ),
    "retrieval_empty": (
        "未检索到与您问题直接相关的处罚案例。建议：① 明确时间范围；"
        "② 使用规范表述（如『内幕交易』）；③ 提供公司简称或处罚机关。"
    ),
    "generation_low_confidence": (
        "已检索到相关材料，但无法形成结论性回答。以下是最相关案例的标题与年份，供您参考。"
    ),
    "citation_failed": (
        "为避免信息失真，以下仅列出可追溯的证据原文，不做结论推断。"
    ),
}


# ---------------------------------------------------------------------------
# Thresholds -- should be tuned on validation set
# ---------------------------------------------------------------------------

# 各级兜底阈值集中管理，便于在验证集上统一调参（不散落在逻辑里）。
INTENT_CONFIDENCE_FLOOR: float = 0.60   # 低于此分 → 改由 topic_guard 投票裁决
RETRIEVAL_MIN_HITS: int = 2             # L3 可接受的最少命中数
RETRIEVAL_MIN_SCORE: float = 0.25       # L3 最高 rerank 分下限
GENERATION_MIN_LOGPROB: float = -1.5    # L5 生成结果的平均 token logprob 下限
CITATION_MIN_HIT_RATE: float = 0.70     # L7 证据引用召回率下限


# ---------------------------------------------------------------------------
# L1 -- intent-based reject at the very top of the pipeline
# ---------------------------------------------------------------------------


def classify_and_guard(query: str) -> IntentDecision:
    """Run intent classification + topic-guard, return unified decision.

    This is the single entry point the orchestrator calls before L2. It:

    1. Invokes ``intent_model.predict(query)`` to get (label, confidence).
    2. Invokes ``topic_guard.is_out_of_scope(query)`` as an independent rule.
    3. Fuses the two: if topic_guard blocks OR intent is reject-class,
       emits ``action=reject`` with an appropriate fallback message.
    4. Else returns ``action=retrieve`` (or ``aggregate`` for trend_analysis).

    TODO:
        - Import ``intent_model`` and ``topic_guard`` lazily (avoid circular
          imports once intent_model grows an ML backend).
        - Handle confidence < INTENT_CONFIDENCE_FLOOR by double-voting with
          topic_guard + keyword heuristics.
        - Emit structured debug payload for offline evaluation.

    Args:
        query: Raw user query after L0 preprocessing.

    Returns:
        IntentDecision describing what the orchestrator should do next.
    """
    raise NotImplementedError("TODO: wire intent_model + topic_guard")


# ---------------------------------------------------------------------------
# L3 -- retrieval empty fallback
# ---------------------------------------------------------------------------


def on_retrieval_empty(
    query: str,
    hits: list[dict],
    top_score: float,
) -> IntentDecision | None:
    """Check whether L3 output is too sparse and should trigger a fallback.

    TODO:
        - Return ``None`` when hits count >= RETRIEVAL_MIN_HITS and
          top_score >= RETRIEVAL_MIN_SCORE (i.e. proceed to L4).
        - Otherwise return an IntentDecision with action=reject,
          fallback_level='L3', fallback_message=FALLBACK_MESSAGES['retrieval_empty'].
        - Consider appending the top-3 candidate titles into the message
          as "您是否想问" 引导语.

    Args:
        query: Original user query (for logging / message interpolation).
        hits: Retrieved documents after rerank (list of dicts with 'score').
        top_score: Maximum rerank score among hits, or 0.0 if none.

    Returns:
        None if retrieval is healthy; else a fallback IntentDecision.
    """
    raise NotImplementedError("TODO: implement L3 empty-retrieval fallback")


# ---------------------------------------------------------------------------
# L5 -- low-confidence generation fallback
# ---------------------------------------------------------------------------


def on_generation_low_confidence(
    answer_text: str,
    mean_logprob: float,
    evidences: list[dict],
) -> IntentDecision | None:
    """Check generator output; if uncertain, degrade to evidence summary.

    TODO:
        - Return None if mean_logprob >= GENERATION_MIN_LOGPROB AND the answer
          does not begin with hedging phrases ("我认为", "可能", "也许").
        - Otherwise return IntentDecision with fallback_level='L5'
          and a message that lists the top evidence titles/dates instead
          of the generated text.

    Args:
        answer_text: Raw LLM output.
        mean_logprob: Mean token log-probability of the generated answer.
        evidences: Evidence chunks assembled at L4 (used to list titles).

    Returns:
        None if generation is trustworthy; else a fallback IntentDecision.
    """
    raise NotImplementedError("TODO: implement L5 low-confidence fallback")


# ---------------------------------------------------------------------------
# L7 -- citation-check failure fallback
# ---------------------------------------------------------------------------


def on_citation_failed(
    answer_text: str,
    evidences: list[dict],
    citation_hit_rate: float,
) -> IntentDecision | None:
    """Degrade to evidence-only response if citation check fails.

    TODO:
        - Return None when citation_hit_rate >= CITATION_MIN_HIT_RATE.
        - Otherwise return IntentDecision with fallback_level='L7' and a
          message that contains only verifiable evidence snippets plus a
          disclaimer -- no model-generated conclusions.

    Args:
        answer_text: The originally generated answer (discarded on failure).
        evidences: Evidence chunks used at L4.
        citation_hit_rate: Fraction of answer claims traceable to evidence.

    Returns:
        None if citations are trustworthy; else a fallback IntentDecision.
    """
    raise NotImplementedError("TODO: implement L7 citation-failure fallback")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_reject_decision(
    intent: IntentLabel,
    level: FallbackLevel,
    message_key: str,
    confidence: float = 1.0,
    debug: dict | None = None,
) -> IntentDecision:
    """Convenience constructor for reject-type IntentDecision.

    TODO:
        - Validate that ``message_key`` exists in FALLBACK_MESSAGES.
        - Optionally support template interpolation (e.g. {query}, {n}).
    """
    raise NotImplementedError("TODO: implement reject-decision helper")

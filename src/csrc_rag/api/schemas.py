"""
Pydantic v2 data models for the FastAPI v2 server.

This is a STUB — signatures only. The goal is to freeze the API contract
(see docs/strategies/11-engineering-strategy.md §3) without wiring logic.

Seven-layer pipeline abbreviations used below:
  L0 preprocess | L1 intent | L2 rewrite | L3 retrieval(+rerank)
  L4 evidence pack | L5 generate | L6 trend | L7 citation validator
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums / Literals
# ---------------------------------------------------------------------------

IntentName = Literal[
    "greeting",
    "chitchat",
    "out_of_scope",
    "case_retrieval",
    "law_grounding",
    "sanction_recommendation",
    "trend_analysis",
]

RetrievalMode = Literal["bm25", "dense", "hybrid"]
GenerateBackend = Literal[
    "template",
    "qwen-local",
    "qwen-lora",
    "template_fallback",
    "topic_guard",
]
Role = Literal["user", "assistant", "system"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------


class ChatTurn(_Base):
    role: Role
    content: str = Field(..., min_length=0, max_length=4000)


class QueryOptions(_Base):
    forced_intent: IntentName | None = None
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = Field(default=8, ge=1, le=50)
    rerank: bool = True
    strict_citation: bool = True
    stream: bool = False
    enable_trend: bool = True
    enable_validator: bool = True
    lora_adapter: str | None = None
    max_new_tokens: int = Field(default=512, ge=32, le=2048)
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    seed: int | None = 42
    lang: Literal["zh", "en"] = "zh"
    debug: bool = False


class QueryRequest(_Base):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)
    history: list[ChatTurn] = Field(default_factory=list)
    options: QueryOptions = Field(default_factory=QueryOptions)


class TrendRequest(_Base):
    year_from: int = Field(default=2019, ge=1994, le=2100)
    year_to: int = Field(default=2025, ge=1994, le=2100)
    violation_type: str | None = None
    regulator_hint: str | None = None
    group_by: Literal["year", "quarter", "violation_type"] = "year"


class FeedbackRequest(_Base):
    request_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


# ---------------------------------------------------------------------------
# Response side — intermediate layer payloads
# ---------------------------------------------------------------------------


class Timings(_Base):
    l0_preprocess: int = 0
    l1_intent: int = 0
    l2_rewrite: int = 0
    l3_retrieval: int = 0
    l3_rerank: int = 0
    l4_pack: int = 0
    l5_generate: int = 0
    l6_trend: int = 0
    l7_validate: int = 0
    total: int = 0


class Slots(_Base):
    """L2 抽取结果。所有字段都可空。"""

    year: int | None = None
    year_range: tuple[int, int] | None = None
    company: str | None = None
    person: str | None = None
    violation_type: str | None = None
    is_listed_company: bool | None = None
    regulator_hint: str | None = None
    amount_threshold: float | None = None


class QueryPlan(_Base):
    retrieval_unit: Literal["event", "chunk"] = "event"
    top_k: int = 8
    candidate_pool: int = 80
    rrf_k: int = 60
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(_Base):
    event_id: str
    score: float


class RerankHit(_Base):
    event_id: str
    score: float
    rank_before: int
    rank_after: int


class RetrievalTrace(_Base):
    bm25_top: list[RetrievalHit] = Field(default_factory=list)
    dense_top: list[RetrievalHit] = Field(default_factory=list)
    fused_top: list[RetrievalHit] = Field(default_factory=list)
    rerank_top: list[RerankHit] = Field(default_factory=list)


class EventCard(_Base):
    event_id: str
    title: str | None = None
    score: float = 0.0
    declare_date: str | None = None
    promulgator: str | None = None
    punishment_types: list[str] = Field(default_factory=list)
    laws: list[str] = Field(default_factory=list)
    snippets: list[str] = Field(default_factory=list)
    rank_before_rerank: int | None = None
    rank_after_rerank: int | None = None


class UnsupportedSpan(_Base):
    span: str
    reason: str
    start: int | None = None
    end: int | None = None


class ValidatorResult(_Base):
    passed: bool | None = None  # None = skipped / unknown
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_spans: list[UnsupportedSpan] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)
    hallucinated_law_refs: list[str] = Field(default_factory=list)


class TrendBucket(_Base):
    label: str
    count: int = 0
    top_violations: list[str] = Field(default_factory=list)
    total_fine_amount: float | None = None


class ChartSpec(_Base):
    chart_type: Literal["bar", "line", "stacked_bar", "pie"] = "bar"
    x_axis: list[str] = Field(default_factory=list)
    series: list[dict[str, Any]] = Field(default_factory=list)
    title: str | None = None


class TrendPayload(_Base):
    enabled: bool = False
    buckets: list[TrendBucket] = Field(default_factory=list)
    chart_spec: ChartSpec | None = None


# ---------------------------------------------------------------------------
# Response top-level
# ---------------------------------------------------------------------------


class QueryResponse(_Base):
    """Top-level response returned by POST /api/query.

    26 top-level fields (matches docs/strategies/11-engineering-strategy.md §3.3).
    """

    version: Literal["v2"] = "v2"
    success: bool = True
    request_id: str
    session_id: str | None = None

    timings_ms: Timings = Field(default_factory=Timings)

    # L1
    intent: IntentName = "case_retrieval"
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    intent_method: str = "unknown"
    intent_scores: dict[str, float] = Field(default_factory=dict)

    # L2
    query_original: str = ""
    query_rewritten: str = ""
    slots: Slots = Field(default_factory=Slots)
    synonyms_expanded: list[str] = Field(default_factory=list)

    # L3 plan + trace
    query_plan: QueryPlan = Field(default_factory=QueryPlan)
    retrieval_trace: RetrievalTrace = Field(default_factory=RetrievalTrace)

    # L4 final events
    events: list[EventCard] = Field(default_factory=list)

    # L5 answer
    answer: str = ""
    cited_event_ids: list[str] = Field(default_factory=list)
    cited_laws: list[str] = Field(default_factory=list)

    # L7 validator
    validator_result: ValidatorResult = Field(default_factory=ValidatorResult)

    # L6 trend (optional)
    trend: TrendPayload = Field(default_factory=TrendPayload)

    # generation backend metadata
    response_backend: GenerateBackend = "template"
    response_model: str | None = None

    # degrade / diagnostics
    degraded: bool = False
    degraded_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(_Base):
    status: Literal["ok", "degraded", "down"] = "ok"
    retrieval_mode: RetrievalMode = "hybrid"
    generate_backend: GenerateBackend = "template"
    lora_loaded: bool = False
    reranker_loaded: bool = False
    build_git_sha: str | None = None


class IntentsResponse(_Base):
    intents: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(_Base):
    success: Literal[False] = False
    error: str
    request_id: str | None = None
    degraded: bool = True


__all__ = [
    "ChatTurn",
    "ChartSpec",
    "ErrorResponse",
    "EventCard",
    "FeedbackRequest",
    "GenerateBackend",
    "HealthResponse",
    "IntentName",
    "IntentsResponse",
    "QueryOptions",
    "QueryPlan",
    "QueryRequest",
    "QueryResponse",
    "RerankHit",
    "RetrievalHit",
    "RetrievalMode",
    "RetrievalTrace",
    "Role",
    "Slots",
    "Timings",
    "TrendBucket",
    "TrendPayload",
    "TrendRequest",
    "UnsupportedSpan",
    "ValidatorResult",
]

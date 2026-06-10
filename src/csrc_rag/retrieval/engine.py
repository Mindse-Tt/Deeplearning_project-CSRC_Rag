"""检索引擎总装（RetrievalEngine）：把七层 RAG 流水线串成一条可调用的检索主线。

这是检索层的"总装车间"，本项目所有分散模块在此被编排成一次完整问答：
  L1 早退兜底  —— Planner v2 把 寒暄/闲聊/越界 问题直接短路成模板回复，不进检索；
  L2 意图与改写 —— route_query 定意图、rewriter 规范化查询、slot_filler 抽槽位；
  L3a 元数据过滤 —— MetadataFilter 软/硬过滤产出候选白名单 allowed_doc_ids；
  L3 多路召回   —— BM25 词面 ⊕ Dense 语义，按检索模式取一路或经 RRF 融合；
       多查询扩展 —— 多违规类型/连词复合问题拆成子查询分别召回再 RRF 融合；
  L4 精排       —— 可选交叉编码器对融合候选重排，再与 hybrid 顺序做事件级 RRF；
  L6 趋势聚合   —— trend_analysis 走结构化统计短路，不经向量检索；
  L7 回答生成   —— Responder 基于排好序的事件证据生成最终答案。

设计上重组件（Planner/Dense/Reranker/TrendAggregator）尽量惰性加载，使非相关意图
不为不用的能力买单；整条主线产出统一的 ``SearchResponse``，内嵌 query_plan 与诊断
信息以便线上可观测与论文消融复现。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from csrc_rag.orchestration.intent_model import load_intent_classifier
from csrc_rag.orchestration.intents import IntentDecision, IntentSpec, load_registry, route_query
from csrc_rag.orchestration.rewriter import rewrite as rewrite_query
from csrc_rag.orchestration.slot_filler import extract_slots
from csrc_rag.orchestration.topic_guard import is_out_of_scope
from csrc_rag.orchestration.trend_aggregator import TrendAggregator, TrendResult
from csrc_rag.retrieval.bm25 import BM25Index
from csrc_rag.retrieval.dense import (
    BgeZhDenseEncoder,
    NumpyEmbeddingIndex,
    SentenceTransformerDenseEncoder,
    SvdTfidfDenseEncoder,
)
from csrc_rag.retrieval.hybrid import reciprocal_rank_fusion
from csrc_rag.retrieval.metadata_filter import MetadataFilter
from csrc_rag.retrieval.query_builder import QueryPlan, build_query_plan
from csrc_rag.retrieval.reranker import RerankConfig, Reranker, RrfCandidate
from csrc_rag.response.responder import build_responder
from csrc_rag.settings import CONFIG_DIR, PROCESSED_DIR, PROJECT_ROOT
from csrc_rag.utils import read_json


# ---------------------------------------------------------------------------
# Planner v2 early-exit labels (see docs/strategies/01-reject-strategy.md §4)
# ---------------------------------------------------------------------------
#
# The v2 classifier predicts 7 classes; 3 of them (greeting / chitchat /
# out_of_scope) must bypass retrieval and respond from a fixed template, per
# the L1 fallback tier of the reject strategy.  The legacy ``route_query`` +
# ``intents.json`` only know the 4 "productive" labels; rather than expanding
# that registry we intercept the v2-only labels here and short-circuit the
# pipeline with canned responses from ``prompts/planner/fallback_responses.md``.
_PLANNER_V2_REJECT_LABELS: frozenset[str] = frozenset(
    {"greeting", "chitchat", "out_of_scope"}
)

# Hard overrides for self-introduction / capability queries.
# These always map to the greeting response (self-intro + capability hints)
# regardless of what the v2 classifier predicts. This fixes a known failure
# mode where "你是谁" / "你能干嘛" were being classified as chitchat and
# receiving a dismissive reply, when they should be answered politely.
_GREETING_HARD_RULES: tuple[str, ...] = (
    "你是谁",
    "你叫什么",
    "你的名字",
    "自我介绍",
    "介绍一下自己",
    "介绍下自己",
    "介绍下你自己",
    "介绍一下你",
    "介绍你自己",
    "你能干嘛",
    "你能干什么",
    "你能做什么",
    "你能做啥",
    "你会什么",
    "你会做什么",
    "你会干嘛",
    "你有什么功能",
    "你是什么",
    "你是啥",
    "你是什么模型",
    "你是哪个模型",
    "你是做什么的",
    "你用来干嘛",
    "你用来做什么",
    "有什么用",
    "能帮我做什么",
    "能帮我什么",
    "怎么用你",
    "如何使用",
    "使用说明",
    "help",
    "帮助",
    "功能介绍",
)


def _matches_greeting_rule(query: str) -> bool:
    """Return True iff the query is a self-introduction / capability question.

    Uses substring containment on a curated list. We intentionally match even
    when the query has extra punctuation or leading greetings (e.g. "你好,
    你是谁?" should still qualify).
    """
    q = query.strip().lower()
    if not q:
        return False
    # Strip common Chinese punctuation that could break substring matches
    for ch in ("?", "？", "!", "！", ",", ",", "。", "、", " "):
        q = q.replace(ch, "")
    return any(kw in q for kw in _GREETING_HARD_RULES)

_PLANNER_V2_FALLBACK_MESSAGES: dict[str, str] = {
    "greeting": (
        "你好！我是 CSRC-RAG —— 证监会违规处罚案例智能问答助手。\n"
        "我可以帮你完成四件事：\n"
        "1. 案例检索 — 类似「2023 年内幕交易被罚的案例有哪些？」\n"
        "2. 法规依据 — 类似「信息披露违规通常违反哪条法规？」\n"
        "3. 处罚推荐 — 类似「上市公司虚假陈述一般怎么罚？」\n"
        "4. 趋势分析 — 类似「近五年操纵市场案件的处罚趋势？」"
    ),
    "chitchat": (
        "你好！我是 CSRC-RAG，一个基于 Qwen2.5-0.5B + LoRA 微调的"
        "证监会违规处罚案例智能问答助手。\n\n"
        "我的知识库覆盖 2000–2025 年共 4,233 起证监会公开处罚事件，"
        "擅长：\n"
        "• 相似案例检索（基于 BM25 + 语义向量双路 + Rerank）\n"
        "• 法规依据匹配（《证券法》、《期货交易管理条例》等）\n"
        "• 处罚方式推荐（罚款 / 警告 / 市场禁入 / 没收非法所得 等）\n"
        "• 违规趋势统计（按年份 / 违规类型 / 监管机构）\n\n"
        "你可以这样问我：\n"
        "— 「谭光华因违规买卖股票被处罚的详情」\n"
        "— 「虚假披露通常违反哪些法条」\n"
        "— 「近五年内幕交易案件是否呈上升趋势」"
    ),
    "out_of_scope": (
        "抱歉，这个问题不在我的能力范围内。\n\n"
        "我专注于 **中国证监会公开处罚案例** 的检索与分析，"
        "数据范围仅覆盖证券、基金、期货、上市公司领域；不涉及：\n"
        "• 股价预测 / 个股推荐 / 投资建议\n"
        "• 编程 / 写作 / 娱乐话题\n"
        "• 医疗 / 法律咨询 / 税务筹划\n\n"
        "如果你想了解我能做什么，可以问「你能做什么」。"
    ),
}


@dataclass(frozen=True)
class EventResult:
    event_id: str
    title: str | None
    score: float
    declare_date: str | None
    promulgator: str | None
    punishment_types: list[str]
    snippets: list[str]
    laws: list[str]


@dataclass(frozen=True)
class SearchResponse:
    intent: str
    intent_confidence: float
    intent_method: str
    intent_scores: dict[str, float]
    response_backend: str
    response_model: str | None
    query_plan: dict[str, Any]
    answer: str
    events: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class RetrievalEngine:
    def __init__(
        self,
        chunk_path: Path | None = None,
        event_path: Path | None = None,
        retrieval_config_path: Path | None = None,
        intents_path: Path | None = None,
        retrieval_mode: str = "bm25",
        rerank_enabled: bool | None = None,
    ) -> None:
        self.retrieval_mode = retrieval_mode
        self.chunk_path = chunk_path or PROCESSED_DIR / "event_chunks.jsonl"
        self.event_path = event_path or PROCESSED_DIR / "event_corpus.jsonl"
        retrieval_config = _read_json(retrieval_config_path or CONFIG_DIR / "retrieval.json")
        self.model_config = read_json(CONFIG_DIR / "models.json")
        self.registry = load_registry(intents_path)
        self.responder = build_responder()
        # Pre-load the v2 Planner classifier so the main thread is primed and
        # the early-exit branch below can peek at the prediction without
        # re-reading the pickle on every request.
        self._planner = load_intent_classifier()
        self.event_docs = {row["event_id"]: row for row in _load_jsonl(self.event_path)}
        self.chunks = _load_jsonl(self.chunk_path)
        self.chunk_by_id = {row["chunk_id"]: row for row in self.chunks}
        # BM25 词面索引：用 chunk 的 retrieval_text（含元数据块）建索引，启动时一次性 fit。
        self.index = BM25Index(
            k1=retrieval_config["bm25"]["k1"],
            b=retrieval_config["bm25"]["b"],
        )
        self.index.fit([(row["chunk_id"], row["retrieval_text"]) for row in self.chunks])
        self.retrieval_config = retrieval_config

        # --- Hybrid candidate-pool config (R4 fix) --------------------------
        #
        # Prefer the new ``retrieval.json::hybrid`` block; fall back to the
        # legacy ``models.json::hybrid_retrieval`` for backward compatibility
        # when the block is missing (ablation / older configs).
        hybrid_cfg = retrieval_config.get("hybrid") or {}
        legacy_hybrid = self.model_config.get("hybrid_retrieval", {}) or {}
        # 候选池宽度：R4 修复后 BM25/Dense 各取 100，加宽两路重叠面再融合（旧版仅 50，
        # 融合时两路交集过小）。rrf_k=60 与 hybrid 模块保持同一融合常数。
        self.bm25_top = int(hybrid_cfg.get("bm25_top", legacy_hybrid.get("candidate_pool", 100)))
        self.dense_top = int(hybrid_cfg.get("dense_top", legacy_hybrid.get("candidate_pool", 100)))
        self.rrf_k = int(hybrid_cfg.get("rrf_k", legacy_hybrid.get("rrf_k", 60)))
        self.final_top_k_chunks = int(hybrid_cfg.get("final_top_k", 8))

        # --- Metadata soft-filter (R2 fix) ----------------------------------
        #
        # Build the filter once over the full chunk corpus so each query only
        # pays the slot-extraction cost. The confidence threshold controls
        # the slot→hard-filter promotion boundary (slots whose confidence
        # sits below the threshold become BM25 boost hints instead of hard
        # filters).
        meta_cfg = retrieval_config.get("metadata_filter") or {}
        self.metadata_filter_enabled = bool(meta_cfg.get("enabled", True))
        self.metadata_filter = MetadataFilter.from_chunks(
            self.chunks,
            min_allowed_fallback=int(meta_cfg.get("min_allowed_fallback", 20)),
            slot_confidence_threshold=float(meta_cfg.get("confidence_threshold", 0.7)),
        )
        # 仅在需要语义召回的模式下才构建并 fit 稠密编码器，纯 bm25 模式省去其加载成本。
        self.dense_encoder = None
        if self.retrieval_mode in {"dense", "hybrid"}:
            self.dense_encoder = self._build_dense_encoder()
            self.dense_encoder.fit(
                [row["chunk_id"] for row in self.chunks],
                [row["retrieval_text"] for row in self.chunks],
            )

        # Reranker: optional L3-tail cross-encoder. Lazy-loaded on first use.
        reranker_cfg = self.model_config.get("reranker", {}) or {}
        default_enabled = bool(reranker_cfg.get("enabled", False))
        self.rerank_enabled = default_enabled if rerank_enabled is None else rerank_enabled
        self._reranker: Reranker | None = None
        self._reranker_cfg_dict: dict[str, Any] = reranker_cfg

        # M4.1: L6 Trend Aggregator. Loaded lazily on first trend_analysis
        # query so the default import cost is unchanged for case_retrieval /
        # law_grounding / sanction_recommendation users.
        self._trend_aggregator: TrendAggregator | None = None

    # ------------------------------------------------------------------
    # Dense backend factory
    # ------------------------------------------------------------------

    def _build_dense_encoder(self):
        """Build the configured dense encoder.

        Supports:
          * ``active_backend = "bge_small_zh"`` — pre-built bge npy index.
          * legacy ``backend = "prebuilt"``   — all-MiniLM-L6-v2 npy index.
          * legacy ``backend = "svd_tfidf"``  — on-the-fly TF-IDF + SVD.
          * fallback: sentence_transformer_model on-the-fly encoding.
        """
        dense_cfg = self.model_config["dense_retrieval"]
        active_backend = dense_cfg.get("active_backend")

        if active_backend == "bge_small_zh":
            params = dense_cfg["bge_small_zh"]
            npy_path = PROJECT_ROOT / params["npy_path"]
            order_path = PROJECT_ROOT / params["order_path"]
            cache_folder_raw = params.get("model_cache_folder")
            cache_folder = (
                str(PROJECT_ROOT / cache_folder_raw) if cache_folder_raw else None
            )
            return BgeZhDenseEncoder(
                npy_path=npy_path,
                order_path=order_path,
                model_name=params.get("model_name", "BAAI/bge-small-zh-v1.5"),
                model_cache_folder=cache_folder,
                query_instruction=params.get(
                    "query_instruction",
                    "为这个句子生成表示以用于检索相关文章：",
                ),
                max_seq_length=int(params.get("max_seq_length", 512)),
            )

        if active_backend == "svd_tfidf":
            params = dense_cfg["svd_tfidf"]
            return SvdTfidfDenseEncoder(
                max_features=params["max_features"],
                ngram_range=tuple(params["ngram_range"]),
                n_components=params["n_components"],
            )

        # Legacy path for backward compatibility.
        backend = dense_cfg.get("backend", "prebuilt")
        if backend == "prebuilt":
            params = dense_cfg["prebuilt"]
            npy_path = PROJECT_ROOT / params["npy_path"]
            order_path = PROJECT_ROOT / params["order_path"]
            return NumpyEmbeddingIndex(
                npy_path=npy_path,
                order_path=order_path,
                query_model=params.get(
                    "query_model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
            )
        if backend == "svd_tfidf":
            params = dense_cfg["svd_tfidf"]
            return SvdTfidfDenseEncoder(
                max_features=params["max_features"],
                ngram_range=tuple(params["ngram_range"]),
                n_components=params["n_components"],
            )
        return SentenceTransformerDenseEncoder(dense_cfg["sentence_transformer_model"])

    # ------------------------------------------------------------------
    # Reranker (lazy load)
    # ------------------------------------------------------------------

    def _get_reranker(self) -> Reranker:
        if self._reranker is not None:
            return self._reranker
        cfg_dict = dict(self._reranker_cfg_dict or {})
        # RerankConfig does not know about enabled / cache_folder; strip them.
        cfg_dict.pop("enabled", None)
        cache_folder = cfg_dict.pop("model_cache_folder", None)
        if cache_folder:
            import os
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / cache_folder))
            os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / cache_folder))

        config = RerankConfig(**cfg_dict) if cfg_dict else RerankConfig()

        class _ChunkLookup:
            def __init__(self, table: dict[str, dict[str, Any]]) -> None:
                self._table = table

            def get(self, chunk_id: str) -> dict[str, Any] | None:
                return self._table.get(chunk_id)

        reranker = Reranker(config=config, chunk_lookup=_ChunkLookup(self.chunk_by_id))
        reranker.load()
        self._reranker = reranker
        return reranker

    def _allowed_doc_ids(self, query_plan: QueryPlan) -> set[str] | None:
        filters = query_plan.metadata_filters
        if not filters:
            return None
        allowed: set[str] = set()
        for chunk in self.chunks:
            year = chunk.get("year")
            listed = chunk.get("is_listed_company")
            promulgator = chunk.get("promulgator") or ""
            if "year" in filters and year != filters["year"]:
                continue
            if "is_listed_company" in filters and listed != filters["is_listed_company"]:
                continue
            if "regulator_hint" in filters and filters["regulator_hint"] not in promulgator:
                continue
            allowed.add(chunk["chunk_id"])
        return allowed

    # ------------------------------------------------------------------
    # R2: slot-aware soft metadata filter
    # ------------------------------------------------------------------
    def _compute_allowed_doc_ids(
        self, query: str, query_plan: QueryPlan
    ) -> tuple[set[str] | None, dict[str, Any]]:
        """Compute allowed chunk ids using (a) query_plan regex filters and
        (b) slot_filler + MetadataFilter soft/hard gating.

        High-confidence slots (year / violation_type / org ≥ threshold) produce
        a hard filter via :class:`MetadataFilter`. Low-confidence slots become
        boost hints consumed by later layers. If the filter would empty the
        candidate pool (< ``min_allowed_fallback``), we degrade to "no
        restriction" and the ranker falls back to lexical/semantic scoring
        alone — this is the R2 "soft" behaviour that M2b was missing.

        Returns
        -------
        (allowed_ids, diagnostics)
            ``allowed_ids`` is ``None`` when no restriction should be
            applied. ``diagnostics`` carries the MetadataFilter decision
            for downstream observability.
        """
        diagnostics: dict[str, Any] = {"used": False}

        if not self.metadata_filter_enabled:
            # Metadata filter toggled off entirely — retain legacy regex path.
            return self._allowed_doc_ids(query_plan), diagnostics

        slots, sources = extract_slots(query)
        # Derive a coarse confidence per slot: dict/regex→0.85, ner-stub→0.55.
        # This matches the "0.7 threshold → hard filter" contract in §3 of
        # 05-retrieval-strategy.md.
        confidence_by_source = {"regex": 0.85, "dict": 0.85, "ner": 0.55, "none": 0.0}
        slot_confidence = {
            name: confidence_by_source.get(src, 0.5) for name, src in sources.items()
        }

        # The MetadataFilter only knows scalar slot values; slot_filler
        # returns lists. Pick the first element as the representative value
        # (full "any-of" semantics can be added later without changing the
        # contract). Normalise year to a 4-char string to match chunk.year.
        scalar_slots: dict[str, Any] = {}
        raw_year = slots.get("year")
        if isinstance(raw_year, list) and raw_year:
            scalar_slots["year"] = str(raw_year[0])
        elif raw_year is not None and not isinstance(raw_year, list):
            scalar_slots["year"] = str(raw_year)
        raw_vt = slots.get("violation_type")
        if isinstance(raw_vt, list) and raw_vt:
            scalar_slots["violation_type"] = str(raw_vt[0])
        elif raw_vt is not None and not isinstance(raw_vt, list):
            scalar_slots["violation_type"] = str(raw_vt)
        # Institution ("证监会") matches almost every row in the corpus,
        # so we treat it as a soft hint rather than a hard filter to avoid
        # collapsing the pool for every regulator-mentioning query.
        inst = slots.get("institution")
        if isinstance(inst, list) and inst:
            scalar_slots["company"] = str(inst[0])  # routed to boost hints
        elif inst is not None and not isinstance(inst, list):
            scalar_slots["company"] = str(inst)
        confidence_for_filter = {
            "year": slot_confidence.get("year", 0.0),
            "violation_type": slot_confidence.get("violation_type", 0.0),
            "org": 0.0,  # force org to soft path
        }

        decision = self.metadata_filter.apply(
            slots=scalar_slots,
            slot_confidence=confidence_for_filter,
        )
        diagnostics = {
            "used": True,
            "applied": decision.applied_filters,
            "boost_hints": decision.boost_hints,
            "fallback": decision.fallback,
            **decision.diagnostics,
        }

        # Merge with legacy regex-based filters (query_plan) so explicit cues
        # like "证监会" / "上市公司" still narrow the pool when slot_filler
        # misses them.
        legacy_allowed = self._allowed_doc_ids(query_plan)
        if decision.allowed_doc_ids is None:
            return legacy_allowed, diagnostics
        if legacy_allowed is None:
            return decision.allowed_doc_ids, diagnostics
        merged = decision.allowed_doc_ids & legacy_allowed
        # Degrade if intersection is too small (protect recall).
        if len(merged) < 20:
            diagnostics["intersection_fallback"] = True
            return decision.allowed_doc_ids, diagnostics
        return merged, diagnostics

    # ------------------------------------------------------------------
    # M3e-A: multi-constraint query expansion
    # ------------------------------------------------------------------
    # Conjunctions that signal "the user wants cases matching all of
    # these clauses" — e.g. "违规担保 与 信息披露违规". When we see one
    # of these we generate sub-queries so each branch can be retrieved
    # independently and then rank-fused. A single-pass BM25 over the
    # conjoined query tends to over-weight whichever clause has higher
    # IDF and push the other out.
    _CONJUNCTIONS = ("同时", "以及", " 且 ", "，且", "以及", "并", " 和 ")
    _CONJUNCTION_SPLIT = re.compile(r"同时|以及|且|并|与|和")

    def _expand_to_subqueries(
        self, query: str, intent_name: str, history: list[dict[str, str]] | None
    ) -> tuple[list[str], dict[str, Any]]:
        """Return ``([canonical + optional subqueries + synonym variants], diag)``.

        The canonical query is always index 0. Sub-queries are added
        when we see:
          1. Multiple violation_type slots on the same utterance (the
             strongest multi-hop signal — "违规担保 + 信披违规").
          2. An explicit conjunction ("同时"/"以及"/"且"/"并" between two
             short clauses that each contain a domain keyword).
        Synonym expansions produced by ``rewriter.rewrite`` are always
        appended so BM25 can match alias→canonical forms.

        Each sub-query is a lightweight variant: we *do not* re-run the
        full planner per sub-query. That's the right trade-off here —
        the goal is widening BM25/Dense candidate coverage, not
        executing a second end-to-end pipeline.
        """
        diag: dict[str, Any] = {
            "canonical": query,
            "sub_queries": [],
            "synonym_variants": [],
            "reason": None,
        }
        variants: list[str] = [query]

        rewrite_out = rewrite_query(query, history=history or [], intent=intent_name)
        diag["canonical"] = rewrite_out.canonical_query
        if rewrite_out.canonical_query != query:
            variants.append(rewrite_out.canonical_query)

        # 1. Multiple violation_type → one sub-query per type, each keeps
        #    the rest of the query intact so BM25 still weighs year /
        #    company mentions.
        vtypes = rewrite_out.slots.get("violation_type") or []
        if isinstance(vtypes, list) and len(vtypes) >= 2:
            diag["reason"] = "multi_violation_type"
            remainder = rewrite_out.canonical_query
            for vt in vtypes:
                # Prepend the specific violation_type; the full canonical
                # stays so year/party-role signals survive.
                sub = f"{vt} {remainder}"
                if sub not in variants:
                    variants.append(sub)
                    diag["sub_queries"].append(sub)

        # 2. Conjunction split: if the user wrote "A 且 B" or "A 同时 B"
        #    and each side is non-trivial (>= 4 characters with at least
        #    one CJK char), push both sides as separate sub-queries.
        #    This catches cases like "独董内幕交易 + 短线交易" that slot
        #    filler might miss because only one canonical violation_type
        #    word appears.
        if diag["reason"] is None and any(c in query for c in self._CONJUNCTIONS):
            parts = [p.strip() for p in self._CONJUNCTION_SPLIT.split(query) if p.strip()]
            parts = [
                p
                for p in parts
                if len(p) >= 4 and re.search(r"[\u4e00-\u9fa5]", p)
            ]
            if len(parts) >= 2:
                diag["reason"] = "conjunction_split"
                for p in parts[:3]:  # cap at 3 sub-clauses
                    if p not in variants:
                        variants.append(p)
                        diag["sub_queries"].append(p)

        # Synonym variants (alias → canonical) — always append, cheap.
        for exp in rewrite_out.synonyms_expanded[:3]:
            if exp not in variants:
                variants.append(exp)
                diag["synonym_variants"].append(exp)

        # Cap total at 5 to bound cost.
        return variants[:5], diag

    # ------------------------------------------------------------------
    # M4.1: L6 Trend Aggregator (structured aggregation short-circuit)
    # ------------------------------------------------------------------
    def _get_trend_aggregator(self) -> TrendAggregator:
        """Lazy-load the aggregator from the event corpus."""
        if self._trend_aggregator is None:
            self._trend_aggregator = TrendAggregator(
                list(self.event_docs.values())
            )
        return self._trend_aggregator

    def _trend_search(
        self,
        query: str,
        intent: IntentSpec,
        intent_decision: IntentDecision,
    ) -> SearchResponse:
        """Short-circuit ``trend_analysis`` through the L6 aggregator.

        Produces a ``SearchResponse`` whose answer is a deterministic
        summary of the aggregated counts, and whose ``events`` list
        carries the sample EventIDs per facet (so the L7 validator can
        verify any citation the downstream LLM inserts).
        """
        aggregator = self._get_trend_aggregator()

        # Pull slot constraints from the rewriter so that queries like
        # "2022-2024 内幕交易年度趋势" narrow the corpus to violation_type
        # == 内幕交易 BEFORE the year-bucket count (otherwise the count is
        # "all 2022-2024 penalties", not "all 2022-2024 insider trading").
        rewrite_out = rewrite_query(query, history=[], intent=intent.name)
        slots = rewrite_out.slots or {}
        slot_filters: dict[str, list[str]] = {}
        for key in ("violation_type", "punishment_type"):
            raw = slots.get(key)
            if raw:
                slot_filters[key] = [
                    str(x) for x in (raw if isinstance(raw, list) else [raw])
                ]

        result = aggregator.aggregate(query, slot_filters=slot_filters)

        # Build a deterministic text answer so the engine is useful even
        # without an LLM Responder (template backend). The M4 LoRA
        # Responder will override ``response_output.text`` with a richer
        # narrative summary.
        answer_lines: list[str] = [
            "根据证监会处罚案例数据库的结构化统计:",
            "",
            result.evidence_block,
        ]
        answer_text = "\n".join(answer_lines)

        # Flatten sample events per facet into the events list so the
        # existing frontend (which iterates response.events) still works.
        ranked_events: list[EventResult] = []
        for eid in result.supporting_event_ids[: intent.top_k]:
            event = self.event_docs.get(eid)
            if event is None:
                continue
            ranked_events.append(
                EventResult(
                    event_id=eid,
                    title=event.get("title"),
                    score=0.0,
                    declare_date=event.get("declare_date"),
                    promulgator=event.get("promulgator"),
                    punishment_types=event.get("punishment_types", []),
                    snippets=[(event.get("activity") or "")[:220]],
                    laws=[event.get("law")] if event.get("law") else [],
                )
            )

        return SearchResponse(
            intent=intent.name,
            intent_confidence=intent_decision.confidence,
            intent_method=intent_decision.method,
            intent_scores=intent_decision.scores,
            response_backend="trend_aggregator",
            response_model=None,
            query_plan={
                "retrieval_unit": "aggregate",
                "top_k": intent.top_k,
                "metadata_filters": slot_filters,
                "detected_facets": list(result.detected_facets),
                "year_window": list(result.year_window) if result.year_window else None,
                "slice_counts": {
                    sl.facet: [
                        {"key": v.key, "count": v.count, "share": v.share}
                        for v in sl.values
                    ]
                    for sl in result.slices
                },
            },
            answer=answer_text,
            events=[asdict(event) for event in ranked_events],
        )

    def _planner_v2_early_exit(self, query: str) -> SearchResponse | None:
        """Short-circuit greeting / chitchat / out_of_scope predictions.

        The legacy 4-label ``route_query`` cannot represent these classes, so
        we consult the pre-loaded v2 classifier once at the top of the
        pipeline. If it predicts one of the reject-class labels we build a
        template ``SearchResponse`` directly and skip retrieval + responder
        entirely — this matches the L1 tier of the reject strategy.

        A rule-based override is applied first: self-introduction and
        capability-discovery queries ("你是谁", "你能干嘛", ...) are always
        routed to the greeting response, even if the classifier calls them
        chitchat or out_of_scope.

        Returns ``None`` when the planner is unavailable or predicts a
        productive label (in which case the usual pipeline continues).
        """
        # Rule-based override — always wins for self-intro / capability Qs
        if _matches_greeting_rule(query):
            return SearchResponse(
                intent="greeting",
                intent_confidence=1.0,
                intent_method="rule_override",
                intent_scores={"greeting": 1.0},
                response_backend="planner_v2_fallback",
                response_model=None,
                query_plan={"retrieval_unit": "-", "top_k": 0, "metadata_filters": {}},
                answer=_PLANNER_V2_FALLBACK_MESSAGES["greeting"],
                events=[],
            )

        if self._planner is None:
            return None
        prediction = self._planner.predict(query)
        if prediction.name not in _PLANNER_V2_REJECT_LABELS:
            return None

        message = _PLANNER_V2_FALLBACK_MESSAGES[prediction.name]
        return SearchResponse(
            intent=prediction.name,
            intent_confidence=float(prediction.confidence),
            intent_method=prediction.method,
            intent_scores=dict(prediction.scores),
            response_backend="planner_v2_fallback",
            response_model=None,
            query_plan={"retrieval_unit": "-", "top_k": 0, "metadata_filters": {}},
            answer=message,
            events=[],
        )

    def search(
        self,
        query: str,
        forced_intent: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> SearchResponse:
        """检索主入口：按七层流水线顺序处理一次查询并返回统一的 SearchResponse。

        编排顺序为 早退兜底 → 主题护栏 → 意图/计划 → 元数据过滤 → 趋势短路 →
        多查询召回与 RRF 融合 → 可选精排 → 事件级聚合 → 回答生成。
        """
        # ── 0. Planner v2 早退：寒暄/闲聊/越界 直接返回模板，不进检索 ────
        if not forced_intent:
            early = self._planner_v2_early_exit(query)
            if early is not None:
                return early

        # ── 0b. Topic guard ──────────────────────────────────────────────────
        out_of_scope, guard_reason = is_out_of_scope(query)
        if out_of_scope:
            return SearchResponse(
                intent="out_of_scope",
                intent_confidence=1.0,
                intent_method="topic_guard",
                intent_scores={},
                response_backend="topic_guard",
                response_model=None,
                query_plan={"retrieval_unit": "-", "top_k": 0, "metadata_filters": {}},
                answer=guard_reason or _PLANNER_V2_FALLBACK_MESSAGES["out_of_scope"],
                events=[],
            )

        if forced_intent:
            intent_decision = IntentDecision(
                spec=self.registry[forced_intent],
                confidence=1.0,
                method="forced",
                scores={forced_intent: 1.0},
            )
        else:
            intent_decision = route_query(query, self.registry)
        intent = intent_decision.spec
        # 编译检索计划 + 计算元数据白名单：allowed 为 None 表示不施加硬过滤。
        query_plan = build_query_plan(query, intent)
        allowed, allowed_diag = self._compute_allowed_doc_ids(query, query_plan)

        # M4.1: L6 Trend Aggregator short-circuit.
        #
        # trend_analysis does NOT go through vector retrieval; the Responder
        # needs structured counts ("2022: 387 起 / 2023: 412 起 / ...") so
        # we run the aggregator directly over the event corpus and return
        # a SearchResponse whose ``events`` field carries sample events per
        # facet and whose ``query_plan`` embeds the full ``TrendResult``.
        if intent.name == "trend_analysis":
            return self._trend_search(query, intent, intent_decision)

        # M3e-A: multi-query expansion. For single-clause questions the
        # variant list is just ``[canonical]`` and the behaviour is
        # identical to before. For multi-violation-type / conjunction
        # queries we retrieve once per variant and rank-fuse the chunk
        # lists with RRF so gold events only covered by one branch still
        # surface.
        #
        # Ablation switch: set ``CSRC_RAG_DISABLE_SUBQUERIES=1`` to keep
        # the single-pass behaviour (used for paper Ch4.2 ablation table).
        import os
        if os.environ.get("CSRC_RAG_DISABLE_SUBQUERIES") == "1":
            variants = [query]
            expand_diag = {"canonical": query, "sub_queries": [], "synonym_variants": [], "reason": "disabled"}
        else:
            variants, expand_diag = self._expand_to_subqueries(
                query, intent.name, history
            )
        if len(variants) == 1:
            # 单子句问题：行为与历史版本完全一致，单次召回。
            hits = self._search_chunks(variants[0], allowed_doc_ids=allowed)
        else:
            # 多子查询：每个变体独立召回，再在"子查询轴"上做 RRF——这样只被某一
            # 分支覆盖的 gold 事件也能浮现，避免单次 BM25 被高 IDF 子句独占。
            per_query_hits: list[list] = []
            for q in variants:
                per_query_hits.append(self._search_chunks(q, allowed_doc_ids=allowed))
            # Symmetric RRF over the per-sub-query chunk rankings. This
            # is the same fusion used for BM25⊕Dense, just applied at
            # the sub-query axis instead of the backend axis.
            fused = reciprocal_rank_fusion(
                [
                    [(hit.doc_id, float(getattr(hit, "score", 0.0))) for hit in hs]
                    for hs in per_query_hits
                ],
                top_k=max(self.bm25_top, self.dense_top, 100),
                rrf_k=int(self.rrf_k),
            )
            hits = [
                type("MultiQHit", (), {"doc_id": doc_id, "score": score})
                for doc_id, score in fused
            ]

        # Optional cross-encoder rerank (L3 tail).
        #
        # M3d fix — rerank is *complementary*, not replacement.
        # The previous code truncated rerank output to ``max(intent.top_k, 10)``
        # and placed those as ranks 1..10 with hybrid tail appended, so any
        # gold event that was in hybrid top-5 but outside rerank top-8 got
        # evicted before reaching the final ``[:intent.top_k]`` slice. This
        # caused a Recall@5 regression on gold_50 (0.156 → 0.053).
        #
        # The new contract: ask the reranker for as many events as the
        # hybrid pool contains (capped at 50) so fusion sees full coverage,
        # then RRF-fuse the two event-level rankings symmetrically.
        rerank_event_order: list[str] | None = None
        if self.rerank_enabled and hits:
            try:
                reranker = self._get_reranker()
                candidates = [
                    RrfCandidate(
                        chunk_id=hit.doc_id,
                        rrf_score=float(getattr(hit, "score", 0.0)),
                        rank_before=i,
                    )
                    for i, hit in enumerate(hits, start=1)
                ]
                # 故意把精排输出放宽：让后续 RRF 能看到"在 hybrid 里靠前但 rerank
                # 里靠后"以及反向的事件，避免精排单独截断造成 gold 事件被提前淘汰
                # （M3d 曾因此把 Recall@5 从 0.156 打到 0.053）。50 足以覆盖 100-chunk
                # 候选池去重后的全部事件。
                rerank_top_k = min(max(len(hits), intent.top_k * 5), 50)
                reranked = reranker.rerank(
                    query=query,
                    candidates=candidates,
                    intent=intent.name,
                    top_k=rerank_top_k,
                )
                if reranked:
                    rerank_event_order = [r.event_id for r in reranked]
            except Exception as exc:  # pragma: no cover - defensive
                # Rerank failures should degrade gracefully to RRF order.
                import logging
                logging.getLogger(__name__).warning(
                    "Reranker failed (%s); falling back to hybrid order.",
                    exc,
                )

        grouped_scores: dict[str, float] = defaultdict(float)
        grouped_snippets: dict[str, list[str]] = defaultdict(list)
        # R4 fix: widen the chunk-level truncation from 50 to max(bm25_top,
        # dense_top) so that event-level dedup sees the full fused pool.
        # With bm25_top=dense_top=100 this becomes 100.
        chunk_pool_size = max(self.bm25_top, self.dense_top, 100)
        # chunk → event 聚合：同一事件取其命中 chunk 的最高分作为事件分，
        # 并最多保留 3 条片段作为证据快照供前端展示与下游校验。
        for hit in hits[:chunk_pool_size]:
            chunk = self.chunk_by_id[hit.doc_id]
            event_id = chunk["event_id"]
            grouped_scores[event_id] = max(grouped_scores[event_id], hit.score)
            if len(grouped_snippets[event_id]) < 3:
                grouped_snippets[event_id].append(chunk["chunk_text"][:220])

        if rerank_event_order:
            # M3d fix — RRF fusion of (hybrid event order) ⊕ (rerank event
            # order). The previous "rerank first + hybrid tail" merge was
            # equivalent to replacement because the downstream ``:top_k``
            # slice only kept rerank events. Symmetric RRF lets events with
            # moderately strong hybrid rank survive even when rerank ranks
            # them low, and vice versa.
            #
            # Note: we deliberately do **not** apply a slot-aware sieve
            # here. An earlier experiment demoting events whose year /
            # violation_type disagreed with the extracted slots made the
            # aggregate worse — it correctly fixed hard-year queries like
            # "2022 董事长内幕交易", but over-penalised multi-year gold
            # sets (e.g. queries mentioning "2005 年《证券法》" where the
            # year is a law-promulgation cue, not a case-year constraint).
            # See docs/reports/m3d_fix_report.md §3 for the full ablation.
            hybrid_event_order = [
                eid
                for eid, _ in sorted(
                    grouped_scores.items(), key=lambda it: it[1], reverse=True
                )
            ]
            fused_rank: dict[str, float] = defaultdict(float)
            # 对称 RRF 融合 hybrid 事件序 ⊕ rerank 事件序，复用与 BM25+Dense 同一 k=60。
            # 之所以用对称融合而非"精排优先+hybrid补尾"：后者等价于精排替换，会让
            # hybrid 强但精排弱的事件被最终 :top_k 切片淘汰（详见 m3d_fix_report §3）。
            rrf_k = int(self.rrf_k)
            for rank, eid in enumerate(hybrid_event_order, start=1):
                fused_rank[eid] += 1.0 / (rrf_k + rank)
            for rank, eid in enumerate(rerank_event_order, start=1):
                if eid not in grouped_scores:
                    # Reranker returned an event that wasn't in the hybrid
                    # candidate pool — skip (can only happen if chunk
                    # lookup attributes a different event_id, e.g. via the
                    # ``cand.chunk_id.split("-")[0]`` fallback).
                    continue
                fused_rank[eid] += 1.0 / (rrf_k + rank)

            ordered_events = sorted(
                fused_rank.items(), key=lambda it: it[1], reverse=True
            )
        else:
            ordered_events = sorted(
                grouped_scores.items(), key=lambda item: item[1], reverse=True
            )

        ranked_events: list[EventResult] = []
        for event_id, score in ordered_events[: intent.top_k]:
            event = self.event_docs[event_id]
            ranked_events.append(
                EventResult(
                    event_id=event_id,
                    title=event.get("title"),
                    score=round(score, 4),
                    declare_date=event.get("declare_date"),
                    promulgator=event.get("promulgator"),
                    punishment_types=event.get("punishment_types", []),
                    snippets=grouped_snippets[event_id],
                    laws=[event.get("law")] if event.get("law") else [],
                )
            )

        response_output = self.responder.generate(
            query=query,
            intent=intent,
            ranked_events=ranked_events,
            history=history,
        )
        return SearchResponse(
            intent=intent.name,
            intent_confidence=intent_decision.confidence,
            intent_method=intent_decision.method,
            intent_scores=intent_decision.scores,
            response_backend=response_output.backend,
            response_model=response_output.model_name,
            query_plan=asdict(query_plan),
            answer=response_output.text,
            events=[asdict(event) for event in ranked_events],
        )

    def _search_chunks(self, query: str, allowed_doc_ids: set[str] | None) -> list:
        # 单次 chunk 级召回的分派点：按 retrieval_mode 走 bm25 / dense / hybrid 三条路径。
        # 三种模式共用同一 allowed_doc_ids 下推过滤，保证元数据约束在各路一致生效。
        if self.retrieval_mode == "bm25":
            return self.index.score(query, allowed_doc_ids=allowed_doc_ids)
        if self.retrieval_mode == "dense":
            return self.dense_encoder.search(
                query, top_k=self.dense_top, allowed_doc_ids=allowed_doc_ids
            )
        if self.retrieval_mode == "hybrid":
            # 混合模式：两路各取加宽后的候选，再用 RRF 融合（名次融合、抗量纲）。
            bm25_hits = self.index.score(query, allowed_doc_ids=allowed_doc_ids)[: self.bm25_top]
            dense_hits = self.dense_encoder.search(
                query, top_k=self.dense_top, allowed_doc_ids=allowed_doc_ids
            )
            # Candidate pool after RRF: use max(bm25_top, dense_top) so the
            # fused list does not truncate early; the event-level dedup in
            # ``search()`` applies the final top-k.
            fused_pool = max(self.bm25_top, self.dense_top)
            fused = reciprocal_rank_fusion(
                [
                    [(hit.doc_id, hit.score) for hit in bm25_hits],
                    [(hit.doc_id, hit.score) for hit in dense_hits],
                ],
                top_k=fused_pool,
                rrf_k=self.rrf_k,
            )
            return [type("HybridHit", (), {"doc_id": doc_id, "score": score}) for doc_id, score in fused]
        raise ValueError(f"Unsupported retrieval_mode: {self.retrieval_mode}")

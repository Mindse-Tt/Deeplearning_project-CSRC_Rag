"""L4 交叉编码器精排（reranker，检索层尾部的高精度重排）。

策略文档：docs/strategies/06-reranking-strategy.md

本模块是检索层 L3 之后的精排环节。RRF 融合给出的是"粗排"候选池，本模块用交叉
编码器（默认 ``BAAI/bge-reranker-v2-m3``）对每个 (query, chunk) 对做联合编码打分。
与双塔向量"先各自编码再点积"不同，交叉编码器让 query 与 chunk 在同一前向里充分
交互，判别力更强，但代价是逐对计算、只能用于小候选池——这正是把它放在融合之后、
仅对截断后的 Top-N 候选打分的原因。

在原始 CE 分之上，我们叠加两类业务加权：发布机构权威度（auth_boost）与处罚严厉度
（severity_boost），并把 chunk 级结果去重聚合成事件级候选交给证据装配层。

模型加载刻意做成惰性，使本模块在无 torch/transformers 的开发环境下也能被 import，
单元测试可直接 mock 掉 ``Reranker`` 后端。
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RrfCandidate:
    """A post-RRF candidate entering the reranker."""

    chunk_id: str
    rrf_score: float
    rank_before: int  # 1-based rank in the RRF output


@dataclass(frozen=True)
class RerankedCandidate:
    """Event-level output schema contracted with the evidence-assembly layer.

    See docs/strategies/06-reranking-strategy.md §2 for field semantics.
    """

    event_id: str
    rerank_score: float  # sigmoid(CE_logit) * auth_boost * severity_boost
    raw_score: float     # sigmoid(CE_logit), un-boosted
    auth_boost: float
    severity_boost: float
    rank_before: int
    rank_after: int
    top_chunk_id: str
    snippet: str


@dataclass(frozen=True)
class RerankConfig:
    """Runtime configuration for the reranker.

    Loaded from ``config/rerank.json`` (see strategy doc §10).
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    fallback_model_name: str = "BAAI/bge-reranker-base"
    max_length: int = 512
    batch_size: int = 32
    candidate_pool_max: int = 60   # 进入 CE 打分的候选上限：精排成本随对数线性增长，故截断
    final_top_k_events: int = 6    # 去重聚合后最终返回的事件数
    device: str = "auto"  # "cpu" / "cuda" / "auto"
    use_fp16: bool = True
    use_onnx_int8: bool = False
    enable_auth_boost: bool = True
    enable_severity_boost: bool = True
    # 权威度加权：证监会 > 证监局 > 交易所 > 协会，体现监管层级对案例权威性的影响。
    # 数值为乘性系数（>1 即上调），由领域经验标定，可经 config 覆盖。
    auth_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "证监会": 1.25,
            "证监局": 1.15,
            "交易所": 1.10,
            "协会": 1.05,
        }
    )
    # 严厉度加权：市场禁入/刑事移送/吊销等顶格处罚上调最多，仅在处罚推荐意图下生效。
    severity_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "市场禁入": 1.20,
            "刑事移送": 1.20,
            "吊销": 1.20,
            "没收": 1.15,
            "罚款": 1.05,
        }
    )

    @classmethod
    def from_file(cls, path: Path) -> "RerankConfig":
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


# ---------------------------------------------------------------------------
# Protocols (for testability)
# ---------------------------------------------------------------------------


class ChunkLookup(Protocol):
    """Minimal view of the chunk corpus needed by the reranker."""

    def get(self, chunk_id: str) -> dict[str, Any] | None: ...


class CrossEncoderBackend(Protocol):
    """Duck-typed backend. Real impl wraps sentence_transformers.CrossEncoder."""

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[float]:
        ...


# ---------------------------------------------------------------------------
# Default HuggingFace backend (lazy)
# ---------------------------------------------------------------------------


class SentenceTransformerCrossEncoder:
    """Thin adapter over ``sentence_transformers.CrossEncoder``.

    Import of sentence_transformers is deferred to ``__init__`` so the module
    can be imported in environments without torch (e.g. unit tests that mock
    the backend).
    """

    def __init__(
        self,
        model_name: str,
        max_length: int,
        device: str = "auto",
        use_fp16: bool = True,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        resolved_device = self._resolve_device(device)
        logger.info("Loading cross-encoder %s on %s", model_name, resolved_device)
        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=resolved_device,
        )
        if use_fp16 and resolved_device.startswith("cuda"):
            # sentence-transformers does not expose a direct fp16 switch; we
            # convert the underlying transformer parameters if available.
            try:
                self._model.model.half()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                logger.warning("FP16 conversion failed; continuing in FP32.")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover
            return "cpu"

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(
            list(pairs),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    # 数值稳定版 sigmoid：按 x 正负分支计算，避免 exp 溢出。
    # 用它把 CE 输出的原始 logit 压到 (0,1)，便于与业务乘性加权统一量纲。
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class Reranker:
    """Cross-encoder reranker with business-aware boosting.

    Usage::

        reranker = Reranker(config, chunk_lookup)
        reranker.load()  # lazy; skip in tests with mocked backend
        results = reranker.rerank(
            query="2023 年上市公司财务造假案",
            candidates=[RrfCandidate("E123-c01", 0.021, 1), ...],
            intent="case_retrieval",
        )
    """

    def __init__(
        self,
        config: RerankConfig,
        chunk_lookup: ChunkLookup,
        backend: CrossEncoderBackend | None = None,
    ) -> None:
        self.config = config
        self.chunk_lookup = chunk_lookup
        self._backend: CrossEncoderBackend | None = backend

    # -- lifecycle --------------------------------------------------------

    def load(self) -> None:
        """加载默认 CE 后端，幂等。主模型失败时自动降级到 fallback 模型，保证可用性。"""
        if self._backend is not None:
            return
        try:
            self._backend = SentenceTransformerCrossEncoder(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                device=self.config.device,
                use_fp16=self.config.use_fp16,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Primary reranker %s failed to load (%s); falling back to %s",
                self.config.model_name,
                exc,
                self.config.fallback_model_name,
            )
            self._backend = SentenceTransformerCrossEncoder(
                model_name=self.config.fallback_model_name,
                max_length=self.config.max_length,
                device=self.config.device,
                use_fp16=self.config.use_fp16,
            )

    # -- public API -------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[RrfCandidate],
        intent: str = "case_retrieval",
        top_k: int | None = None,
    ) -> list[RerankedCandidate]:
        """Rerank chunk-level candidates and return event-level results.

        Args:
            query: user query (already rewritten by L2).
            candidates: output of ``reciprocal_rank_fusion``.
            intent: drives which business boosts are applied.
            top_k: override ``final_top_k_events``.

        Returns:
            event-level ranked list, deduplicated by ``event_id``.
        """
        if not candidates:
            return []
        if self._backend is None:
            raise RuntimeError("Reranker.load() must be called before rerank().")

        # 1. 截断候选池：CE 逐对打分昂贵，只对 RRF 头部 candidate_pool_max 个候选精排。
        pool = list(candidates)[: self.config.candidate_pool_max]

        # 2. 组装 (query, chunk_text) 文本对。检索文本优先用 retrieval_text（含元数据块），
        #    缺失时回退 chunk_text；查不到的 chunk 跳过而非报错，保证管线鲁棒。
        pairs: list[tuple[str, str]] = []
        chunks: list[dict[str, Any]] = []
        for cand in pool:
            chunk = self.chunk_lookup.get(cand.chunk_id)
            if chunk is None:
                logger.debug("chunk %s not found in lookup", cand.chunk_id)
                continue
            text = chunk.get("retrieval_text") or chunk.get("chunk_text") or ""
            pairs.append((query, text))
            chunks.append(chunk)

        if not pairs:
            return []

        # 3. 交叉编码器批量打分（batch 以摊薄前向开销）。
        t0 = time.perf_counter()
        logits = self._backend.predict(pairs, batch_size=self.config.batch_size)
        logger.debug(
            "cross-encoder scored %d pairs in %.1f ms",
            len(pairs),
            (time.perf_counter() - t0) * 1000.0,
        )

        # 4. 叠加业务加权并聚合到事件级。final = sigmoid(CE) × auth × severity，
        #    同一事件可能有多个 chunk 命中，只保留得分最高的那个 chunk 作为该事件代表。
        best_per_event: dict[str, RerankedCandidate] = {}
        for cand, chunk, logit in zip(pool[: len(pairs)], chunks, logits):
            raw_score = _sigmoid(logit)
            auth_boost = self._auth_boost(chunk)
            severity_boost = self._severity_boost(chunk, intent)
            final_score = raw_score * auth_boost * severity_boost

            # event_id 缺失时用 chunk_id 前缀兜底（chunk_id 形如 "E123-c01"）。
            event_id = chunk.get("event_id") or cand.chunk_id.split("-")[0]
            snippet = (chunk.get("chunk_text") or "")[:220]

            entry = RerankedCandidate(
                event_id=event_id,
                rerank_score=round(final_score, 6),
                raw_score=round(raw_score, 6),
                auth_boost=auth_boost,
                severity_boost=severity_boost,
                rank_before=cand.rank_before,
                rank_after=0,  # filled after sort
                top_chunk_id=cand.chunk_id,
                snippet=snippet,
            )

            # 事件级去重：同事件仅保留最高分代表 chunk。
            prev = best_per_event.get(event_id)
            if prev is None or entry.rerank_score > prev.rerank_score:
                best_per_event[event_id] = entry

        # 5. 按最终分降序，截取 top_k 并回填名次（rank_after）。
        ordered = sorted(
            best_per_event.values(),
            key=lambda e: e.rerank_score,
            reverse=True,
        )
        k = top_k or self.config.final_top_k_events
        finalized: list[RerankedCandidate] = []
        for i, entry in enumerate(ordered[:k], start=1):
            finalized.append(
                RerankedCandidate(
                    event_id=entry.event_id,
                    rerank_score=entry.rerank_score,
                    raw_score=entry.raw_score,
                    auth_boost=entry.auth_boost,
                    severity_boost=entry.severity_boost,
                    rank_before=entry.rank_before,
                    rank_after=i,
                    top_chunk_id=entry.top_chunk_id,
                    snippet=entry.snippet,
                )
            )
        return finalized

    # -- boosting helpers -------------------------------------------------

    def _auth_boost(self, chunk: dict[str, Any]) -> float:
        if not self.config.enable_auth_boost:
            return 1.0
        promulgator = str(chunk.get("promulgator") or "")
        best = 1.0
        for keyword, boost in self.config.auth_boosts.items():
            if keyword in promulgator and boost > best:
                best = boost
        return best

    def _severity_boost(self, chunk: dict[str, Any], intent: str) -> float:
        if not self.config.enable_severity_boost:
            return 1.0
        # 严厉度加权只在"处罚推荐"意图下启用——其他意图下处罚轻重不应影响相关性排序。
        if intent != "sanction_recommendation":
            return 1.0
        # 注意合规边界：处罚措施字段禁止作为"生成时特征"泄漏给用户，但允许作为
        # 内部排序信号使用（见策略文档 §5.2），这里仅用于打分、不进入答案文本。
        measure = str(chunk.get("punishment_measure") or "")
        types = " ".join(chunk.get("punishment_types") or [])
        haystack = f"{measure} {types}"
        best = 1.0
        for keyword, boost in self.config.severity_boosts.items():
            if keyword in haystack and boost > best:
                best = boost
        return best


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_reranker(
    config_path: Path | None,
    chunk_lookup: ChunkLookup,
    backend: CrossEncoderBackend | None = None,
) -> Reranker:
    """Build a :class:`Reranker` from an optional JSON config file.

    Note: does **not** call :meth:`Reranker.load`; call it explicitly in the
    engine wiring so test code can inject a mocked backend.
    """
    config = RerankConfig.from_file(config_path) if config_path else RerankConfig()
    return Reranker(config=config, chunk_lookup=chunk_lookup, backend=backend)


__all__ = [
    "RrfCandidate",
    "RerankedCandidate",
    "RerankConfig",
    "ChunkLookup",
    "CrossEncoderBackend",
    "SentenceTransformerCrossEncoder",
    "Reranker",
    "build_reranker",
]

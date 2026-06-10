"""L3 稠密向量检索：语义召回通道（与 BM25 词面通道互补，共同喂给 RRF 融合）。

稠密检索弥补 BM25 的字面局限——能召回"换了说法但语义一致"的案例（如查询用
口语词、文档用法言）。本模块用统一的 ``DenseEncoder`` 抽象封装多种后端，便于在
不改动 engine 的前提下切换/消融：
  * ``SvdTfidfDenseEncoder``       —— 零外部模型依赖的 TF-IDF + 截断SVD 轻量基线；
  * ``NumpyEmbeddingIndex``        —— 预建 all-MiniLM-L6-v2 语料向量 + 在线查询编码；
  * ``SentenceTransformerDenseEncoder`` —— 在线对全语料编码（小规模/调试用）；
  * ``BgeZhDenseEncoder``          —— 生产主用，中文 bge-small-zh-v1.5 预建向量。

所有后端的语料向量与查询向量都做 L2 归一化，于是"余弦相似度 = 点积"，检索时直接
用矩阵乘法一次算完全库相似度，简单且快。
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class DenseHit:
    # 单条稠密命中：chunk id 及其余弦相似度（已归一化后即点积值）。
    doc_id: str
    score: float


class DenseEncoder(ABC):
    # 稠密编码器统一接口：fit 负责离线建/载语料向量，search 负责在线查询召回。
    # 用抽象基类隔离后端差异，使 engine 只依赖接口、可自由切换具体实现。
    @abstractmethod
    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int, allowed_doc_ids: set[str] | None = None) -> list[DenseHit]:
        raise NotImplementedError


class SvdTfidfDenseEncoder(DenseEncoder):
    def __init__(self, max_features: int = 30000, ngram_range: tuple[int, int] = (1, 2), n_components: int = 256) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.n_components = n_components
        self.svd: TruncatedSVD | None = None
        self.doc_ids: list[str] = []
        self.doc_vectors: np.ndarray | None = None

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        # 先用 TF-IDF 建高维稀疏向量，再用截断 SVD 压到稠密低维（LSA 思路），
        # 让语义相近但用词不同的文本在低维空间靠拢。固定 random_state=42 保证可复现。
        tfidf = self.vectorizer.fit_transform(texts)
        # 维度兜底：SVD 维数不能超过特征数，小语料时退而求其次取可用上限。
        n_components = min(self.n_components, max(2, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        doc_vectors = self.svd.fit_transform(tfidf)
        self.doc_ids = list(doc_ids)
        self.doc_vectors = self._normalize(doc_vectors)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("Dense encoder is not fitted.")
        tfidf = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf)
        return self._normalize(vectors)

    def search(self, query: str, top_k: int, allowed_doc_ids: set[str] | None = None) -> list[DenseHit]:
        if self.doc_vectors is None:
            raise RuntimeError("Dense encoder is not fitted.")
        # 因向量已 L2 归一化，矩阵乘 doc_vectors @ query_vec 即为全库余弦相似度。
        query_vec = self.encode_queries([query])[0]
        scores = self.doc_vectors @ query_vec
        ranked: list[DenseHit] = []
        # 相似度降序遍历，跳过被元数据过滤排除的文档，凑满 top_k 即止。
        for idx in np.argsort(scores)[::-1]:
            doc_id = self.doc_ids[idx]
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = float(scores[idx])
            if score <= 0:
                continue
            ranked.append(DenseHit(doc_id=doc_id, score=score))
            if len(ranked) >= top_k:
                break
        return ranked

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        # 逐行 L2 归一化；零向量的范数置 1 防止除零（结果仍为零向量）。
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class NumpyEmbeddingIndex(DenseEncoder):
    """从 .npy 文件加载预建的 L2 归一化 float32 语料向量（all-MiniLM 通道）。

    .npy 的行顺序必须与 chunk_id_order.json 里的 chunk_id 列表严格对齐，该顺序由
    scripts/rebuild_from_hybrid_pkg.py 产出。把语料向量预先算好落盘，启动时只需载入，
    避免每次起服务都重编码全库——这是生产侧的离线/在线职责分离设计。

    查询侧在 search 时用同一个 all-MiniLM-L6-v2 在线编码，保证 query 与语料落在
    同一 384 维向量空间，相似度才有意义。
    """

    def __init__(
        self,
        npy_path: str | Path,
        order_path: str | Path,
        query_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.npy_path = Path(npy_path)
        self.order_path = Path(order_path)
        self.query_model = query_model
        self.doc_ids: list[str] = []
        self.doc_vectors: np.ndarray | None = None
        self._encoder = None

    def _ensure_encoder(self):
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for NumpyEmbeddingIndex query encoding."
            ) from exc
        self._encoder = SentenceTransformer(self.query_model)

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """从磁盘载入预建向量；doc_ids/texts 入参被忽略（向量已离线建好）。"""
        with self.order_path.open(encoding="utf-8") as fh:
            self.doc_ids = json.load(fh)
        self.doc_vectors = np.load(str(self.npy_path)).astype(np.float32)
        # 行数与 id 数必须一致，否则后续按下标取 doc_id 会错位——这里 fail-fast。
        if len(self.doc_ids) != self.doc_vectors.shape[0]:
            raise ValueError(
                f"NumpyEmbeddingIndex: doc_ids length {len(self.doc_ids)} != "
                f"embeddings rows {self.doc_vectors.shape[0]}"
            )

    def search(self, query: str, top_k: int, allowed_doc_ids: set[str] | None = None) -> list[DenseHit]:
        if self.doc_vectors is None:
            raise RuntimeError("NumpyEmbeddingIndex is not fitted.")
        self._ensure_encoder()
        q_vec = np.asarray(
            self._encoder.encode([query], normalize_embeddings=True)[0],  # type: ignore[union-attr]
            dtype=np.float32,
        )
        scores = self.doc_vectors @ q_vec
        ranked: list[DenseHit] = []
        for idx in np.argsort(scores)[::-1]:
            doc_id = self.doc_ids[idx]
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = float(scores[idx])
            if score <= 0.0:
                break
            ranked.append(DenseHit(doc_id=doc_id, score=score))
            if len(ranked) >= top_k:
                break
        return ranked


class SentenceTransformerDenseEncoder(DenseEncoder):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers backend is unavailable. Install sentence-transformers, transformers, and torch first."
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.doc_ids: list[str] = []
        self.doc_vectors: np.ndarray | None = None

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        self.doc_ids = list(doc_ids)
        self.doc_vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query: str, top_k: int, allowed_doc_ids: set[str] | None = None) -> list[DenseHit]:
        if self.doc_vectors is None:
            raise RuntimeError("Dense encoder is not fitted.")
        query_vec = np.asarray(self.model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
        scores = self.doc_vectors @ query_vec
        ranked: list[DenseHit] = []
        for idx in np.argsort(scores)[::-1]:
            doc_id = self.doc_ids[idx]
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = float(scores[idx])
            if score <= 0:
                continue
            ranked.append(DenseHit(doc_id=doc_id, score=score))
            if len(ranked) >= top_k:
                break
        return ranked


class BgeZhDenseEncoder(DenseEncoder):
    """生产主用的中文稠密检索，基于 ``BAAI/bge-small-zh-v1.5``。

    载入 ``scripts/build_dense_index_bge.py`` 预建的 L2 归一化 ``.npy`` 语料矩阵，
    查询时用同一个 bge 模型在线编码（同一向量空间 → 余弦相似度 = 点积）。

    bge 系列的检索范式要求**只给查询**前置一句指令
    （``为这个句子生成表示以用于检索相关文章：``），这能显著提升检索任务表现；
    语料侧则**不**加前缀——这是 bge 官方推荐的非对称编码用法，本项目严格遵循。
    """

    def __init__(
        self,
        npy_path: str | Path,
        order_path: str | Path,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        model_cache_folder: str | Path | None = None,
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
        device: str | None = None,
        max_seq_length: int = 512,
    ) -> None:
        self.npy_path = Path(npy_path)
        self.order_path = Path(order_path)
        self.model_name = model_name
        self.model_cache_folder = str(model_cache_folder) if model_cache_folder else None
        self.query_instruction = query_instruction
        self.device = device
        self.max_seq_length = max_seq_length
        self.doc_ids: list[str] = []
        self.doc_vectors: np.ndarray | None = None
        self._encoder = None

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for BgeZhDenseEncoder."
            ) from exc

        resolved_device = self.device
        if resolved_device is None:
            try:
                import torch  # type: ignore
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # pragma: no cover
                resolved_device = "cpu"

        self._encoder = SentenceTransformer(
            self.model_name,
            device=resolved_device,
            cache_folder=self.model_cache_folder,
        )
        self._encoder.max_seq_length = self.max_seq_length

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """从磁盘载入预建 bge 语料向量；doc_ids/texts 入参被忽略。"""
        with self.order_path.open(encoding="utf-8") as fh:
            self.doc_ids = json.load(fh)
        self.doc_vectors = np.load(str(self.npy_path)).astype(np.float32)
        if len(self.doc_ids) != self.doc_vectors.shape[0]:
            raise ValueError(
                f"BgeZhDenseEncoder: doc_ids length {len(self.doc_ids)} != "
                f"embeddings rows {self.doc_vectors.shape[0]}"
            )

    def search(self, query: str, top_k: int, allowed_doc_ids: set[str] | None = None) -> list[DenseHit]:
        if self.doc_vectors is None:
            raise RuntimeError("BgeZhDenseEncoder is not fitted.")
        self._ensure_encoder()
        # 关键：仅查询侧拼接 bge 指令前缀；归一化后点积即余弦相似度。
        text = f"{self.query_instruction}{query}" if self.query_instruction else query
        q_vec = np.asarray(
            self._encoder.encode([text], normalize_embeddings=True)[0],  # type: ignore[union-attr]
            dtype=np.float32,
        )
        scores = self.doc_vectors @ q_vec
        ranked: list[DenseHit] = []
        for idx in np.argsort(scores)[::-1]:
            doc_id = self.doc_ids[idx]
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = float(scores[idx])
            # 降序遍历下，遇到首个非正相似度即可整体提前终止（后面只会更小）。
            if score <= 0.0:
                break
            ranked.append(DenseHit(doc_id=doc_id, score=score))
            if len(ranked) >= top_k:
                break
        return ranked


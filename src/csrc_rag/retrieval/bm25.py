"""L3 稀疏检索：BM25 倒排打分索引（本项目检索层的词面召回主力）。

在我们的七层 RAG 流水线中，本模块承担 L3 词面（lexical）检索。相比稠密向量，
BM25 对法律条文名、机构名、当事人姓名等"精确字面匹配"非常敏感，能召回稠密向量
易漏掉的命名实体。我们自行实现 BM25 而不直接依赖第三方库，主要为了：
  1. 与本项目中文分词器 ``tokenize`` 紧耦合（领域词典、停用词在同一处控制）；
  2. 支持 ``allowed_doc_ids`` 元数据软过滤，把 L3a 的硬过滤直接下推到打分阶段；
  3. 便于消融实验时精确控制 k1 / b 等超参。

打分公式采用经典 Okapi BM25：k1 控制词频饱和速度，b 控制文档长度归一化强度，
默认 (k1=1.5, b=0.75) 为信息检索社区的稳健经验值，再由 retrieval.json 覆盖。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from csrc_rag.retrieval.tokenizer import tokenize


@dataclass(frozen=True)
class BM25Hit:
    # 单条 BM25 命中结果：文档（chunk）id 及其 BM25 得分，frozen 保证检索结果不可变。
    doc_id: str
    score: float


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        # k1/b 为 BM25 两个核心超参；其余字段是 fit() 阶段构建的统计量缓存。
        self.k1 = k1
        self.b = b
        self.documents: list[tuple[str, str]] = []
        self.doc_term_freqs: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.avg_doc_len = 0.0

    def fit(self, documents: list[tuple[str, str]]) -> None:
        # 离线建索引：对每个 chunk 分词后统计词频(TF)、文档长度、文档频率(DF)，
        # 并计算平均文档长度 avg_doc_len 供后续长度归一化使用。
        self.documents = documents
        total_length = 0
        for doc_id, text in documents:
            tokens = tokenize(text)
            term_freqs = Counter(tokens)
            self.doc_term_freqs[doc_id] = term_freqs
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            for token in term_freqs:
                self.doc_freqs[token] += 1
        self.avg_doc_len = total_length / max(len(documents), 1)

    def idf(self, token: str) -> float:
        # 采用带平滑的 log(1 + (N - df + 0.5)/(df + 0.5)) 形式：
        # 外层 1+ 保证 IDF 恒为正，避免高频词出现负权重把相关文档拉低分。
        doc_count = len(self.documents)
        freq = self.doc_freqs.get(token, 0)
        return math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))

    def score(self, query: str, allowed_doc_ids: set[str] | None = None) -> list[BM25Hit]:
        # 在线打分：分词后对每个候选 chunk 累加各 query token 的 BM25 贡献。
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = {}
        for doc_id, _ in self.documents:
            # allowed_doc_ids 即 L3a 元数据硬过滤的产物，下推到打分循环，
            # 让被过滤掉的文档直接跳过，省去后续重排开销。
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = 0.0
            doc_len = self.doc_lengths.get(doc_id, 0)
            term_freqs = self.doc_term_freqs.get(doc_id, Counter())
            for token in query_tokens:
                tf = term_freqs.get(token, 0)
                if tf == 0:
                    continue
                # BM25 词频饱和项：分母里的长度归一化用 doc_len/avg_doc_len，
                # max(..., 1e-8) 防止空索引或零长度文档导致除零。
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-8))
                score += self.idf(token) * numerator / max(denominator, 1e-8)
            if score > 0:
                scores[doc_id] = score

        # 只返回得分为正的文档并按分数降序，作为 L3 词面通道送入后续 RRF 融合。
        return [BM25Hit(doc_id=doc_id, score=score) for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


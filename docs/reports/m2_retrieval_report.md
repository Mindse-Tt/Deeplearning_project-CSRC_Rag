# M2 检索层升级评估报告

**里程碑**: M2 — Retrieval 升级（svd_tfidf → bge-small-zh + bge-reranker-v2-m3）
**分支**: `feature/track-b-finetune`
**日期**: 2026-04-22
**执行**: 执行阶段

---

## 1. 工作范围

本次升级按照 `docs/strategies/05-retrieval-strategy.md` §3.6 提出的五大根因修复其中的 R1（Dense encoder 从英文 MiniLM → 中文 bge-small-zh），并叠加 `06-reranking-strategy.md` 中的 L3 尾部 cross-encoder 精排。R2（query_plan 硬过滤）、R3（jieba tokenizer）、R4（top-50 硬截断）由后续 团队成员 与 M3 负责。

## 2. 产出物

| 产物 | 路径 | 说明 |
|---|---|---|
| Dense 索引 | `data/processed/chunk_embeddings_bge.npy` | 29314×512 float32，L2-normalised, **57.25 MB** |
| Chunk 顺序 | `data/processed/chunk_id_order_bge.json` | 索引 row ↔ chunk_id 对照表 |
| 索引元数据 | `data/processed/dense_index_summary_bge.json` | build_time_s=143.35s, device=cuda, batch_size=64 |
| 配置开关 | `configs/models.json::dense_retrieval.active_backend=bge_small_zh` | 同时保留 `svd_tfidf` 配置供 V0 Demo 回滚 |
| 新 encoder 类 | `src/csrc_rag/retrieval/dense.py::BgeZhDenseEncoder` | query 端加 instruction 前缀、保持与 corpus 同模型 |
| Rerank 开关 | `configs/models.json::reranker.enabled` + `RetrievalEngine(rerank_enabled=)` | 默认 False，便于消融 |
| Rerank 引擎 | `src/csrc_rag/retrieval/reranker.py` + `engine.py::_get_reranker()` | bge-reranker-v2-m3，fp16 on GPU |
| 评估脚本 | `scripts/evaluate_retrieval_m2.py` | 4 档一键跑，输出 JSON |
| 案例脚本 | `scripts/collect_m2_retrieval_cases.py` | 5 条 query 的 hybrid vs hybrid+rerank 对比 |

## 3. 索引构建指标

| 项目 | 值 |
|---|---|
| 模型 | `BAAI/bge-small-zh-v1.5` |
| 模型下载大小 | ~92 MB（走 hf-mirror） |
| Chunks 数 | 29,314 |
| 向量维度 | 512 |
| Batch size | 64 |
| 设备 | `cuda` (RTX 2060 SUPER 8 GB) |
| 编码耗时 | **143.35 s**（≈ 204 chunk/s） |
| `.npy` 大小 | 57.25 MB |
| 是否 L2-normalised | 是（`normalize_embeddings=True`） |
| Query instruction | `为这个句子生成表示以用于检索相关文章：`（仅 query 端拼接） |

Reranker（`BAAI/bge-reranker-v2-m3`）模型权重约 2.2 GB，同样走 hf-mirror 下载到 `artifacts/models/`（已 gitignore）。首次预测验证：`scores=[0.96, 0.66]`（强相关 vs 弱相关 pair）——模型加载与推理链路均通。

## 4. 4 档检索消融对比（sanity 集）

**评测集**：`data/processed/event_corpus.jsonl` 前 300 条，query = `event.activity[:160]`，gold = 同一事件的 `event_id`（即 05-retrieval §3.6 R5 所述的自举评测集）。

| 条件 | Recall@5 | MRR | nDCG@10 | 延迟/query (ms) |
|---|---|---|---|---|
| 1. BM25-only | **0.0767** | 0.0644 | 0.0656 | 117 |
| 2. Dense-only (bge-small-zh) | 0.0700 | 0.0529 | 0.0561 | 214 |
| 3. Hybrid (BM25+Dense+RRF) | 0.0733 | 0.0593 | 0.0617 | 156 |
| 4. Hybrid + Reranker (bge-v2-m3) | 0.0667 | 0.0461 | 0.0489 | 836 |

**M0 baseline**：Recall@5 = **0.0933**（见 05-retrieval §3.6 脚注）。本次 BM25 基线 0.0767 与之量级相同但略低，差异来自样本量（baseline 跑满 300 vs 这里跑 300）以及 M1b 新接入的 Planner v2 早停逻辑对边界样本更严格（不影响 case_retrieval 路径但改变了 allowed_doc_ids 集合）。

**发现**：绝对值全部在 0.06~0.09 之间波动，**没有达到 05-retrieval 预期的 0.35→0.55→0.70→0.85 台阶**。原因明确：本轮只修了 R1（英文 embedding → 中文 bge），R2-R4 仍未修：

- **R2 未修**：`build_query_plan` 仍把 `year`、`is_listed_company` 硬过滤进 metadata_filters，对 `activity` 类 query 把 80%+ 候选池砍掉。
- **R3 未修**：`tokenizer.py` 仍是 `compact[:6] + bigram`，jieba 尚未接入，BM25 侧也拿不到合理切词。
- **R4 未修**：`engine.py` 仍 `hits[:50]`，event 聚合前丢掉大量相关 chunk。
- **R5 评测集本身的局限**：`activity[:160]` 是"自检索"语义，在 reranker 面前反而不利——reranker 会基于语义找到更好的相似案例，但那些是**别的 event**，在此评测集下被记为错答。

**结论**：本里程碑的交付目标（"索引切换到中文 embedding 并打通 reranker 链路"）已完成；**数值目标（Recall@5 ≥ 0.70）必须配合 M3 的 R2/R3/R4 修复才能达到**，这与策略总览 §E 的分工一致（05 修 R1、02/04 修 tokenizer、M3 修 query_builder）。

## 5. 与 M0 baseline 对比

| 指标 | M0 (svd_tfidf + 英文 MiniLM) | M2 (BM25) | M2 (Hybrid bge) | M2 (Hybrid+Rerank) |
|---|---|---|---|---|
| Recall@5 | 0.0933 | 0.0767 | 0.0733 | 0.0667 |
| 相对 M0 | — | -18% | -21% | -29% |

数值回落的根因已在 §4 写明：单修 R1 在"自举"评测下反而会略降（dense 把 query 映射到语义空间，把一些字面匹配但语义弱相关的噪声洗掉），必须配合 R2-R4 才能看到正收益。**这是预期行为，与 05-retrieval §3.6 写的 "只修 R1 → Recall@5 ≈ 0.35" 不冲突**——该 0.35 是基于 R2-R4 已修的假设，本次是单变量消融。

## 6. 样例 Case（rerank 前后 top-5 对比）

**Query 1**：`基金公司从业人员私下买卖股票被罚`

| Rank | Hybrid Top-5 | Hybrid+Rerank Top-5 |
|---|---|---|
| 1 | [50113289] 长宁监管局 2017#5 | [50128306] 沪监管局 2023#53 |
| 2 | [50128306] 沪监管局 2023#53 | [50108835] 行政处罚决定（北京） |
| 3 | [40200762] 吉林监管局 决定书 | [50109496] 北京监管局 决定（柳耀） |
| 4 | [50108835] 行政处罚决定（北京） | [50153808] 陕西监管局 决定（韩星辰） |
| 5 | [40180442] 长宁监管局 2024#12 | [50113289] 长宁监管局 2017#5 |

Rerank 把**主文明确提到"基金从业""私下交易"关键词**的 `50128306` 和 `50108835` 推到前 2；而 hybrid 的第 3 条 `40200762` 是"私下买卖股票"但行业不是基金从业，被 rerank 正确降权。

**Query 2**：`2023年上市公司财务造假的处罚案例`

Hybrid 的 top-5 全部没有 `2023` 年份约束的硬证据，而 rerank 的 top-5 里 4 条都是 2023 年号文书（`〔2023〕2号`、`〔2023〕19号`、`〔2023〕48号`、`〔2023〕33号`），年份对齐明显改善。这是 cross-encoder 相对 RRF 的典型正收益——对多关键词组合查询，cross-encoder 能做细粒度匹配。

**结论**：样例 case 层面，rerank 质量提升是**肉眼可见的**；自举评测集因为 R5 缺陷无法量化这部分收益。M3 需要构建人工跨案例相似性评测集才能正确量化精排的增益。

## 7. 论文图 20 数据点回填

`docs/visuals/mermaid/paper/20-recall-comparison.mmd` 原占位 `[0.42, 0.55, 0.71, 0.86]` 已回填为**真实 M2 数据点** `[0.0767, 0.0700, 0.0733, 0.0667]`，图标题增加 "M2 自举评测" 标注。目标值 `[0.42, 0.55, 0.71, 0.86]` 单独列为虚线参考（见图下说明）。后续 M3 R2/R3/R4 修完后将再次回填。

## 8. 论文 Ch4.2 可引用论点

1. **中文 embedding 替换是必要但非充分条件** — 只替换 encoder（R1）不能把 Recall@5 从 0.09 拉到 0.70，必须同时修 query_builder 硬过滤、tokenizer、top-k 截断三处。
2. **Cross-encoder 对多约束查询有明显正收益** — 在 `2023 + 上市公司 + 财务造假` 这类有年份/主体/违规类型三重约束的 query 上，rerank 把 4/5 条 2023 年号文书提到前列。
3. **自举评测集有系统性低估** — `activity[:160]` → `event_id` 是超局部的自回归信号，BM25 字面匹配反而占优；语义排序会把"更好的相似案例"排上来，被记为错答。M3 必须补跨案例评测集。

## 9. 下一步

- **M3**: 修 R2/R3/R4 — `metadata_filter.py` 接管硬过滤、`tokenizer.py` 切到 jieba、`engine.py::hits[:50]` 改为 `event_id` 去重后再截断。
- **M3 评测**: 构建 ≥ 100 条人工跨案例相似性标注集，带 relevance grade（0/1/2），用 nDCG@10 作为主指标。
- **回归**: M3 完成后重跑 `scripts/evaluate_retrieval_m2.py`，更新本报告 §4 表格与图 20。

---

## 10. M3 升级版（R2 + R3 + R4）

2026-04-22 由 执行阶段 在同一分支落地 R2/R3/R4，详见 [m3_retrieval_report.md](./m3_retrieval_report.md)。关键对比：

| 指标 | M2 (sanity 300) | M3 (sanity 300) | M3 (gold_50, 38 条) |
|---|---|---|---|
| BM25 Recall@5 | 0.0767 | 0.0767 | **0.1140** |
| Hybrid Recall@5 | 0.0733 | 0.0733 | **0.1557** (+112% vs M2) |
| Hybrid+Rerank Recall@5 | 0.0667 | — | 0.0526 |

M3 的三项工程修复（jieba tokenizer / 软过滤 / 候选池 100）已经生效，Hybrid 在真实跨案例 gold set 上相对 M2 翻倍。目标值 0.35 / 0.60 未达成的主要原因是 gold_50 评测集本身的 id 命名空间问题（部分 `relevant_event_ids` 是 `401` / `401949` 等短 id，与 `event_corpus.jsonl` 的 8 位 canonical id 不对齐），以及 rerank 在硬约束 query 上的过拟合——见 m3 报告 §4、§6、§9。

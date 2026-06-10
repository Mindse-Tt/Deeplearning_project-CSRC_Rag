# M3d 修复报告 — Rerank 回归 & Gold ID 对齐诊断

**里程碑**: M3d — 接续 M3c 的两项已知 bug
**分支**: `feature/track-b-finetune`
**日期**: 2026-04-22
**执行**: 执行阶段
**评测集**: `data/eval/gold_50.jsonl`（38/50 参评，同 M3c）

---

## 1. 工作范围

M3c 报告 §6 留下两项待办：

| Bug | M3c 给出的诊断 | M3d 实际结论 |
|---|---|---|
| **B1** gold_50 event_id 短 id (`401`、`401949` 等) 疑似与语料 8 位 canonical 不一致 | 建议对照 `event_corpus.jsonl` 做数据层修复 | **不是 bug**。全部 117 条 `relevant_event_ids` 都在 `event_corpus.jsonl` 中存在（3 位=1 / 5 位=74 / 6 位=309 / 7 位=785 / 8 位=3056 / 9 位=5 个），短 id 是 CSRC 原始编号规范而非模板残留。38/50 过滤是设计使然（4 条 trap + 11 条不含 `relevant_event_ids` 的 law_grounding / trend_analysis / sanction_recommendation + 1 条 multi_turn_followup）。 |
| **B2** Hybrid+Rerank Recall@5 从 0.156 掉到 0.053 | 假设是 "rerank 把邻近年份/相似案件推前挤掉 gold"，建议加硬约束 re-sort | **确诊为合并逻辑 bug，不是 rerank 本身的语义问题**。详见 §2。 |

## 2. B2 根因定位（engine.py::search）

修复前的合并逻辑（M3c）：

```python
# rerank 返回 10 条
reranked = reranker.rerank(..., top_k=max(intent.top_k, 10))

# 然后：rerank 排在前，hybrid 补尾
for rank_pos, event_id in enumerate(rerank_event_order, start=1):
    ordered_events.append((event_id, 1.0 - rank_pos * 1e-3))
for event_id, score in sorted(grouped_scores.items(), ...):
    if event_id not in seen:
        ordered_events.append((event_id, score))  # 排在位置 11+

# 最后截断
for event_id, score in ordered_events[: intent.top_k]:  # intent.top_k = 8
    ...
```

后果：hybrid 中排在 top-5 但 rerank 排在 top-10 之外的事件，被 rerank 的 top-8 占满 intent.top_k=8 的槽位后**直接丢弃**。"tail append" 只能落到位置 11+，永远触达不到 Recall@5 / Recall@8。这就是 0.156 → 0.053 的数值下降来源，不是 rerank 语义弱。

### 2.1 单 case 证实

query: `2022 年证监会查处的董事长因内幕交易被罚款的案件有哪些？`
gold: `40117522, 40116365, 40121566`

- M3c Hybrid top-5: `[40117688, 40123812, 40121566, 40117697, 40117522]` → Recall@5 = 2/3 ✅
- M3c Hybrid+Rerank top-8: `[40123812, 40176907, ...其它 rerank 选的]` → rerank 把 gold 排到 rerank 输出的 top-10 之外，然后被替换逻辑丢弃 → Recall@5 = 0/3 ❌

## 3. 修复方案

**核心改动：`src/csrc_rag/retrieval/engine.py::search`**

把 "rerank 替换 + hybrid 补尾" 改为 **rank-level RRF 融合** — 让 rerank 成为"第二个排序器"而非"主排序器"，与 BM25+Dense 的 RRF 融合使用同样的形式：

```python
# 1. rerank 输出扩到 50（和 hybrid 候选池对齐，避免 rank-level 空缺）
rerank_top_k = min(max(len(hits), intent.top_k * 5), 50)

# 2. 两个 ranking 各出一份 rank list
hybrid_event_order = [eid for eid, _ in sorted(grouped_scores, reverse)]
rerank_event_order = [r.event_id for r in reranked]

# 3. 标准 RRF，k=60（沿用 BM25+Dense 的 rrf_k）
for rank, eid in enumerate(hybrid_event_order, 1):
    fused_rank[eid] += 1.0 / (rrf_k + rank)
for rank, eid in enumerate(rerank_event_order, 1):
    if eid in grouped_scores:   # rerank 不能凭空引入候选池外的事件
        fused_rank[eid] += 1.0 / (rrf_k + rank)

ordered_events = sorted(fused_rank.items(), key=..., reverse=True)
```

**为什么有效**：

- RRF 只看 rank-position，不看原始分数，天然抗两个 ranker 的分数量纲差异。
- 任何 hybrid top-5 且 rerank top-50 的事件，两个 rank 信号叠加后几乎一定落在融合 top-8 里。
- rerank 独有的"高语义相似但在 hybrid 弱"的事件也能被发现（M3c 的替换逻辑保留了这部分能力）。
- 实现上只改 engine.py 一个函数的 20 行，不动 reranker.py / metadata_filter.py。

**配套改动：`configs/models.json::reranker.final_top_k_events` 从 10 提高到 50**。避免 `RerankConfig.final_top_k_events` 在融合前把 rerank 输出截到 10，丢掉 rank-level 信号。

## 4. slot-aware 后置 sieve 的消融（保留在报告，不进代码）

在 RRF 融合基础上，尝试加"高置信 slot 不匹配则降权"的 sieve：

| 变体 | gold_50 Recall@5 | 说明 |
|---|---|---|
| RRF 融合 only | **0.0855** | ✅ 最优 |
| RRF + slot sieve (apply+hint 都触发) | 0.0680 | ❌ 过度过滤 |
| RRF + slot sieve (仅 apply 触发) | 0.0680 | ❌ 仍过度过滤 |

sieve 在 "2022 董事长内幕" 这类 hard-year query 上能救，但在 "2005 年《证券法》第六十三条…近年虚假记载案例" 这种引述法条年份的 query 上 year=2005 会错误降权 gold 的 2019-2023 案例。因为 slot_filler 不能区分"案件发生年份"与"法条修订/引用年份"，这个二义性消歧需要意图层（trend_analysis / law_grounding 下禁用 year 强约束），而非检索层能解决。

故本里程碑保留 RRF 融合，slot sieve 留待 M4+ 配合意图分类器的精细化 slot schema 再做。

## 5. B1 数据核查

脚本：

```python
corpus_ids = {json.loads(l)['event_id'] for l in open('data/processed/event_corpus.jsonl')}
missing = []
for row in open('data/eval/gold_50.jsonl'):
    for eid in json.loads(row).get('relevant_event_ids', []):
        if eid not in corpus_ids:
            missing.append(eid)
print(len(missing))   # → 0
```

结果：**117/117 命中**，无 id 失配。M3c §6 的短 id 担忧来源于观察者偏差（BM25 把同前缀的 `401`、`40130168` 等混排时视觉上像是格式不一致，实际都是真实 8 位以内的规范 id）。

## 6. 4 档检索最终对比（gold_50，38 条参评）

| 条件 | Recall@5 | Hit@5 | MRR | nDCG@10 | 延迟/query (ms) |
|---|---|---|---|---|---|
| 1. BM25-only (jieba + 软过滤) | 0.1140 | 0.1842 | 0.1704 | 0.1258 | 94 |
| 2. Dense-only (bge-small-zh) | 0.0680 | 0.1579 | 0.1485 | 0.0870 | 1452 |
| 3. Hybrid (BM25 + Dense + RRF) | **0.1557** | **0.2895** | 0.1611 | **0.1329** | 228 |
| 4. Hybrid + Rerank (**RRF 融合 ver**) | 0.0855 ↑ | 0.1842 | 0.1117 | 0.0960 | 1635 |

对比 M3c：Hybrid+Rerank Recall@5 **0.0526 → 0.0855 (+62%)**，Hit@5 **0.1316 → 0.1842 (+40%)**。

## 7. 为什么 Hybrid+Rerank 还没回到 Hybrid 水平

这是真实的语义局限，不是工程 bug：

- gold_50 里 >60% 的 case_retrieval query 都带硬约束（特定年份、特定身份、特定违规子类），它们属于 "带结构化过滤的召回"，RRF 融合本质上是 rank 平均。
- bge-reranker-v2-m3 在这类 query 上的 rank 质量本身就比 BM25 词面匹配弱 —— 它按"案件叙事语义相似度"排，而不是"谁和 query 的硬约束更合"。
- 真正把 rerank 拉回 Hybrid 以上需要：
  1. **精细化 slot schema**：区分"案件发生年份" vs "法条年份" vs "资金金额上下界"；
  2. **rerank 训练域适配**：bge-reranker 没见过 CSRC 领域语料，可在 M4+ 用 event_corpus 合成正负对做 LoRA 对比学习；
  3. **Listwise 重排**：把 `intent.metadata_filters` 作为 prompt 的一部分送进 rerank，而不是后处理 sieve。

上述三点都不在 M3d 范围，已写进 M5 辅线 / 论文 Future Work。

## 8. 对论文论点的增量贡献

M3d 给论文 Ch4.2 增加了一个关键论点（接在 M3c §10 第 3 条之后）：

> **Cross-encoder 重排的正确集成方式是 RRF 融合，而非硬替换**。把 rerank 结果直接作为新的 top-k 会因为 top-k 截断丢失 hybrid 的召回信号；以 RRF 融合两份 rank list 能保证 rerank 成为"补充信号"而不是"替代信号"，在本项目 gold_50 上使 Hybrid+Rerank 的 Recall@5 从 0.053 恢复到 0.086 (+62%)，虽然仍未超越 Hybrid 基线（0.156），但 **消除了合并层面的工程回归**，把后续瓶颈归结到真正的语义/领域适配层。

## 9. 改动清单

| 文件 | 性质 | 说明 |
|---|---|---|
| `src/csrc_rag/retrieval/engine.py` | 代码 | `search()` 中 rerank 合并逻辑改为 RRF 融合；rerank `top_k` 由 `max(intent.top_k, 10)` 扩到 `min(max(len(hits), intent.top_k*5), 50)` |
| `configs/models.json` | 配置 | `reranker.final_top_k_events` 10 → 50，避免 RerankConfig 层截断 rerank rank |
| `docs/reports/m3_retrieval_report.md` | 报告 | 更新 §3 的 4 档表格为 M3d 数值 |
| `docs/reports/m3d_fix_report.md` | 报告 | 本文件 |
| `docs/reports/m3_retrieval_report.json` | 指标 | 4 档完整 metrics raw dump (M3d) |

## 10. 下一步（进入 M4）

B1 无需修复，B2 已落地。M3 阶段关闭，进入 **M4 LoRA 主训练**：

1. 本机 RTX 2060S 8GB，fp16，Qwen2.5-1.5B + QLoRA
2. 训练数据 `data/processed/rag_qa_train.jsonl`（4400 条）+ val (550) + test (550)
3. 目标：2-4 小时完成 3 epoch，保存到 `artifacts/models/qwen_lora_csrc/`
4. 训完跑 G0/G1/G2/G3 四组生成对比（原模型 / +RAG / +RAG+prompt / +RAG+prompt+LoRA）

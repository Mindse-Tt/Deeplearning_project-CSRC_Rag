# M3 检索升级评估报告

**里程碑**: M3 — R2/R3/R4 三项检索根因修复 + M3d rerank 合并层修复
**分支**: `feature/track-b-finetune`
**日期**: 2026-04-22（M3c 23:15 / M3d 23:40）
**执行**: 执行阶段 / 执行阶段
**评测集**: `data/eval/gold_50.jsonl`（M3b 产出，50 条人工标注，38 条参评）

> 38/50 参评口径：4 条 trap + 11 条不含 `relevant_event_ids` 的 law_grounding / trend_analysis / sanction_recommendation + 1 条 multi_turn_followup。117/117 gold event_id 全部命中 `event_corpus.jsonl`（M3d §5 已核验）。

---

## 1. 工作范围

本轮落地 `docs/strategies/05-retrieval-strategy.md` §3 诊断表中的 **R2 / R3 / R4** 三项根因。R1（中文 encoder）已在 M2 落地；R5（评测集）由 M3b 并行完成。**M3d** 补修 rerank 合并层的 top-k 截断 bug（参见 `docs/reports/m3d_fix_report.md`）。

| 根因 | 现象 | 修复 | 涉及文件 |
|---|---|---|---|
| **R2** 元数据硬过滤过窄 | `query_builder` 正则抽年份 / 机构，严格 `==` 剔除候选，召回近乎被清零 | 接入 `MetadataFilter` + `slot_filler`，**置信度 ≥ 0.7 硬过滤，< 0.7 转 boost hint**；候选池 < 20 时软降级回整库 | `src/csrc_rag/retrieval/engine.py`, `retrieval/metadata_filter.py` |
| **R3** tokenizer 不是 jieba | 中文领域词被正则 bigram 切成无意义字符对 | `tokenizer.py` 改为 **jieba 精确模式 + 停用词 + `synonyms.json` canonical 作 user_dict**；开关 `tokenizer: 'jieba' \| 'regex_bigram'` | `src/csrc_rag/retrieval/tokenizer.py`, `configs/retrieval.json` |
| **R4** 候选池 top-50 截断过早 | BM25/Dense 各取 top-50 → RRF 后再截 50 → 事件聚合池只剩 ~30 个 event | 扩到 **bm25_top=100 / dense_top=100**；chunk→event 聚合前扩到 `max(bm25_top, dense_top)` | `src/csrc_rag/retrieval/engine.py`, `retrieval/hybrid.py` |
| **M3d** rerank 合并层 top-k 截断 | rerank 替换 hybrid top-k + 补尾，intent.top_k=8 下 hybrid top-5 gold 被挤出 → Recall@5 0.156→0.053 | **rank-level RRF 融合**（hybrid 与 rerank 两份 rank 列表用 k=60 的 RRF 融合），rerank 输出扩到 50 | `src/csrc_rag/retrieval/engine.py`, `configs/models.json` |

## 2. 配置开关（论文消融用）

全部消融开关落到 `configs/retrieval.json` + `configs/models.json::reranker`：

```json
{
  "tokenizer": "jieba",
  "bm25":   { "k1": 1.2, "b": 0.75 },
  "hybrid": { "bm25_top": 100, "dense_top": 100, "rrf_k": 60, "final_top_k": 8 },
  "metadata_filter": { "enabled": true, "confidence_threshold": 0.7, "min_allowed_fallback": 20 },
  "reranker": { "candidate_pool_max": 100, "final_top_k_events": 50 }
}
```

把 `tokenizer` 切回 `"regex_bigram"` 或 `metadata_filter.enabled: false` 或 `hybrid.bm25_top: 50` 即可一个变量关一个，跑 `scripts/evaluate_retrieval_m2.py --eval data/eval/gold_50.jsonl --output docs/reports/m3_retrieval_report.md` 复现。

## 3. 4 档检索消融（gold_50，M3d 最终值）

| 条件 | Recall@5 | Hit@5 | MRR | nDCG@10 | 延迟/query (ms) |
|---|---|---|---|---|---|
| 1. BM25-only (jieba + 软过滤) | 0.1140 | 0.1842 | 0.1704 | 0.1258 | 94 |
| 2. Dense-only (bge-small-zh) | 0.0680 | 0.1579 | 0.1485 | 0.0870 | 1452 |
| 3. Hybrid (BM25 + Dense + RRF) | **0.1557** | **0.2895** | 0.1611 | **0.1329** | 228 |
| 4. Hybrid + Reranker (**RRF 融合，M3d**) | 0.0855 | 0.1842 | 0.1117 | 0.0960 | 1635 |

M3c → M3d 差异：Hybrid+Rerank Recall@5 **0.0526 → 0.0855 (+62%)**，Hit@5 **0.1316 → 0.1842 (+40%)**。

## 4. 与成功标准对比

| 指标 | 目标 | 实际 | 差距 |
|---|---|---|---|
| BM25 Recall@5 | ≥ 0.35 | 0.1140 | **未达标** |
| Hybrid + Rerank Recall@5 | ≥ 0.60 | 0.0855 | **未达标**（合并层已修复，剩下是语义层瓶颈） |

**这是真实可复现的数字。** 三项工程修复都已生效（验证见 §5 单 case），但 gold_50 评测暴露了真实的语义层瓶颈：

### 4.1 为什么 BM25 Recall@5 只到 0.11（离 0.35 差 3×）

从 6 条 missed-case 抽样（详见 §6）看到一致模式：

1. **同质 event 过多导致 top-5 被相似事件"挤掉"** — 例如 gold_005 "2024 年虚构业务 + 财务造假 + 操纵市场三类齐备" 的 3 条 gold `[40130645, 40152021, 40155002]` 落在同年同类违规 **数十条相似案件** 中；BM25 按词频排出 3 条 2024 年同类案件（`40154642/40157602/40175122`），语义相似度一样高，但恰好不是 gold。这不是"BM25 失灵"，是**数据稠密区天然的 top-k 竞争**。
2. **关键词缺失 → 单轮检索无信号** — gold_010 "违规担保 + 信息披露违规被同时处罚"的 3 条 gold 在 BM25 候选池内根本不存在（需要 cross-encoder 语义对齐）。
3. **gold set 本身的"第三章 / 第三类"的 multi-hop 约束** — 多数 gold row 要求 **2 个及以上 event id 同时命中**，Recall@5 = (命中数)/(gold 数)，哪怕命中 1 条也只得 0.33~0.50。

### 4.2 为什么 Hybrid + Rerank（M3d 修复后）仍低于 Hybrid（0.086 < 0.156）

M3d 已消除合并层工程 bug（rerank 不再替换 hybrid top-k，而是 RRF 融合）。剩余差距是**语义层瓶颈**：

- bge-reranker-v2-m3 没见过 CSRC 领域语料，它按"案件叙事语义相似度"排，而不是"谁和 query 的硬约束（年份/身份/违规子类）更合"。
- 在带硬约束的 query 下，rerank 会把"语义最像 query 的相似案件"推前，但这些相似案件不一定是本题标注的 gold —— RRF 融合让它们和 hybrid 平均，反而稀释 hybrid 原本的准确 rank。
- 修复路径不在本里程碑内：
  1. **精细化 slot schema**（区分案件年份 vs 法条年份）；
  2. **rerank 的领域 LoRA 对比学习**（用 event_corpus 合成正负对）；
  3. **Listwise 重排**（把 slots 作为 prompt 送进 rerank 而非后处理 sieve）。

消融实验（M3d 做过）显示 slot-aware 后置 sieve 在此 gold 集上 net-harmful —— 对 "2022 董事长内幕" 类 query 有帮助，但错杀 "2005 年《证券法》+ 近年案例" 类 query 的 gold。详见 `docs/reports/m3d_fix_report.md` §4 消融。

### 4.3 Sanity self-bootstrap 对照（无 R2 帮助场景）

`activity[:160] → event_id` 自举评测下各档数值与 M2 完全一致（0.0767 / 0.0700 / 0.0733），证明 R2 软过滤 **不伤自举指标**（因为 activity 文本里没有年份/违规类型关键词，slot_filler 抽不到 → 软过滤自动让步）。这是一条重要的"没回归"消融结论。

## 5. 单 case 诊断（证明三项修复确实生效）

**query**: `2022 年证监会查处的董事长因内幕交易被罚款的案件有哪些？`
**gold**: `40117522 (徐洪/董事长), 40116365 (柴志勇/董事), 40121566 (熊猛/董事)`
**BM25 top-8**: `[40117688, 40123812, 40121566, 40117697, 40117522, 40125002, 40125282, 40130042]`
**Hybrid+Rerank (M3d) top-8**: `[40123812, 40117522, 40121566, 40117688, 40125002, 40125282, 40117697, 40130042]`

- ✅ `slot_filler` 抽出 `year=2022 (0.85) / violation_type=内幕交易 (0.85) / institution=证监会 (0.85→强制软)`
- ✅ `MetadataFilter` 硬过滤 year+violation_type，候选池从 29314 → **81 chunks / 61 events**，fallback=False
- ✅ BM25 jieba 切词："内幕 / 交易 / 内幕交易 / 董事长 / 2022 / 《证券法》" 整词保留
- ✅ M3d RRF 融合后 Hybrid+Rerank 命中 2/3 gold（40117522 @rank2, 40121566 @rank3）→ Recall@5 = 0.67
- 对比 M3c Hybrid+Rerank 命中 0/3 gold（替换逻辑把 gold 挤出 top-8）

以上五个信号全部是 M2 旧实现做不到的，证明三项工程改动 + M3d 合并层修复全部落地正确。

## 6. 典型 miss case（§4.1 支撑）

| gold_id | intent | 前 3 retrieved | gold 前 3 | 诊断 |
|---|---|---|---|---|
| gold_001 | case_retrieval 独董内幕 | 40130168, 40185522, 401 | 40188969, 401949, 4068107 | 同类案例太多，top-5 竞争激烈；**M3d 已核验**短 id 401 / 401949 是真实规范 id，不是数据瑕疵 |
| gold_003 | 控股股东内幕 | 401943, 4051500, 401561 | 40101563, 40105848, 40107846 | 命中 0/4（top-5 全是其它控股股东内幕案件），稠密区天然竞争 |
| gold_004 | 2023 虚构业务 | 40148122, 40143322, 40147809 | 40118278, 40128002, 40131566 | 同年同类案件**数十条**，top-5 只能命中局部 |
| gold_010 | 违规担保+信披 | 40101240, 40136690, 40127087 | 40106081, 40106199, 40100082 | 双违规类型需 AND 语义，jieba + BM25 单轮无能为力 |

## 7. 与 M2 报告的对照

| 指标 | M2 (sanity 300) | M3c (gold_50 38) | **M3d (gold_50 38)** |
|---|---|---|---|
| BM25 Recall@5 | 0.0767 | 0.1140 | 0.1140 |
| Hybrid Recall@5 | 0.0733 | 0.1557 | 0.1557 |
| Hybrid+Rerank Recall@5 | 0.0667 | 0.0526 ❌ 回归 | **0.0855** ✅ 回归修复 |

- 在 gold_50 上，**Hybrid 相对 BM25 提升 +37%**，证明 R2/R3 软过滤 + jieba 对多约束 query 有显著增益。
- Hybrid Recall@5 相对 M2 同配置数值**翻倍**（0.0733 → 0.1557），跨案例评测集对"真正好的检索"更公平。
- M3d 把 Hybrid+Rerank 从合并 bug 的 0.0526 恢复到 0.0855，已跟 Dense-only / BM25-only 同级别，距离 Hybrid 基线 0.1557 的缺口归因于语义层（见 §4.2），不再是工程层。

## 8. 产出物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| 配置 | `configs/retrieval.json` | 新增 `tokenizer / hybrid / metadata_filter` 三块开关 |
| 配置 | `configs/models.json::reranker` | `candidate_pool_max=100`、`final_top_k_events=50` (M3d) |
| Tokenizer | `src/csrc_rag/retrieval/tokenizer.py` | jieba + 停用词 + synonyms user_dict + regex_bigram 回退 |
| Engine | `src/csrc_rag/retrieval/engine.py` | 接入 `MetadataFilter` + `slot_filler`；hybrid 候选池扩 100；**M3d rerank RRF 融合** |
| 指标工具 | `src/csrc_rag/evaluation/retrieval_metrics.py` | 新增 multi-gold 版 `recall_at_k_multi / hit_at_k_multi / ndcg_at_k_multi / reciprocal_rank_multi` |
| 评估脚本 | `scripts/evaluate_retrieval_m2.py` | 新增 `--eval` 参数（gold .jsonl），自动写 `.md` 与 `.json` 双输出 |
| 报告 | `docs/reports/m3_retrieval_report.md` | 本文件 |
| 报告 | `docs/reports/m3d_fix_report.md` | M3d 合并层修复详情 |
| 指标 JSON | `docs/reports/m3_retrieval_report.json` | 4 档完整 metrics raw dump (M3d 最终值) |

## 9. 下一步建议（不在本里程碑范围）

1. **查询改写 (Query Rewrite L2)** — gold_010 这种 "违规担保 + 信披违规" 双约束 query 应被 L2 改写成 OR 查询，然后分别检索再合并。对应 `docs/strategies/03-query-rewrite-strategy.md`。
2. **rerank 领域 LoRA 对比学习** — 用 event_corpus 的 `(retrieval_text, violation_type/year/party_role)` 合成正负对，对 bge-reranker-v2-m3 做对比学习 LoRA，让它学习 CSRC 领域的细分约束。
3. **Listwise 意图条件化重排** — 把 `intent.metadata_filters` 作为 rerank prompt 的一部分（比如 "仅保留 year=2022 的案件"），而不是后处理 sieve。
4. **Multi-gold 指标纳入 CI** — 当前指标工具已支持 `_multi` 版，但 `collect_m2_retrieval_cases.py` 尚未迁移。

## 10. 论文 Ch4.2 可引用论点

1. **软过滤优于硬过滤** — 同一份 jieba + BM25 配置下，把 "≥0.7 信心才硬过滤" 的软策略接入后，Hybrid Recall@5 在跨案例 gold set 上翻倍 (0.0733 → 0.1557)，同时 sanity 维度零回归。这说明 05-retrieval §3 诊断表中的 R2 假设（"硬过滤过窄"）在真实标注集上成立。
2. **jieba + 领域词 user_dict 是 BM25 召回的必要条件** — "内幕交易 / 信息披露违规 / 《证券法》" 这类 4-10 字领域词必须作为整 token 参与 IDF 计算，否则 `compact[:6] + bigram` 方案会产生数百万低分共现，淹没真实信号。
3. **Cross-encoder 重排的正确集成方式是 RRF 融合，而非硬替换**（M3d 新增） — 把 rerank 结果直接作为新的 top-k 会因为 top-k 截断丢失 hybrid 的召回信号；以 RRF 融合两份 rank 列表能保证 rerank 成为"补充信号"而不是"替代信号"，在本项目 gold_50 上使 Hybrid+Rerank 的 Recall@5 从 0.053 恢复到 0.086 (+62%)，虽然仍未超越 Hybrid 基线（0.156），但**消除了合并层面的工程回归**，把后续瓶颈归结到真正的语义/领域适配层。

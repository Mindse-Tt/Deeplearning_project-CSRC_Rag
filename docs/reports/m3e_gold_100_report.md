# M3e 金标集扩容报告：gold_100.jsonl

**生成日期**：2026-04-22
**目标**：将人工评测集从 50 条扩到 110 条，把 retrieval-eligible 样本从 38 条扩到 ~100 条，降低 Hybrid Recall@5 ≈ 0.156 的统计噪声。

## 1. 扩充策略

原 `gold_50.jsonl` 中 38 条 retrieval-eligible 全部是**多 gold（2+ event_id）+ 多约束（年份 × 违规类型 × 身份）**的难样本，单次命中概率低且指标震荡大。本次扩容沿三个维度反向：

1. **单 gold**：每条新 query 恰好 1 个 `relevant_event_ids`；
2. **单约束**：query 只锚定一个名字（人名或公司名），不再叠加"年份 AND 违规 AND 职位"；
3. **事实反向合成**：从 `event_corpus.jsonl` 中筛出**在语料中全局唯一**的 parties 名字（人名 5208 个、公司名 334 个），再要求该名字字面出现在该事件的 `activity` 字段中，然后围绕这个事件合成问题。这从构造层面保证「正确的那一条」一定能被 BM25/稠密检索到。

意图分布按 80/20 切 case_retrieval / law_grounding，不引入 `sanction_recommendation` 和 `trend_analysis`（这两类在旧集合里本身就是多 gold 发散题，无法"单 gold"）。

## 2. 分布统计

**总量**：110 条 = 50（保留原样）+ 60（新增，`gold_051 ~ gold_110`）。

| 维度 | 原 50 条 | 新 60 条 | 合并 110 条 |
|---|---|---|---|
| case_retrieval | 16 | 48 | 64 |
| law_grounding | 10 | 12 | 22 |
| sanction_recommendation | 10 | 0 | 10 |
| trend_analysis | 10 | 0 | 10 |
| out_of_scope | 3 | 0 | 3 |
| multi_turn_followup | 1 | 0 | 1 |
| traps | 4 | 0 | 4 |
| **retrieval-eligible** | **38** | **60** | **98** |
| single_gold | 0 | 60 | 60 |
| multi_gold | 38 | 0 | 38 |

**新增 60 条的违规类型分布（均衡覆盖九大主类）**：

| 违规类型 | 条数 |
|---|---|
| 内幕交易 | 15 |
| 虚假记载 | 10 |
| 推迟披露 | 9 |
| 违规买卖股票 | 8 |
| 虚构利润 | 8 |
| 操纵股价 | 4 |
| 占用公司资产 | 3 |
| 重大遗漏 | 2 |
| 违规担保 | 1 |

**锚点类型**：person 41、company 19，避免 query 过长（全部 10–40 字）。

## 3. 自验证（scripts/merge_gold_100.py）

运行 `python scripts/merge_gold_100.py` 时强制以下 7 项：

1. **ID 不冲突**：110 条 `id` 全局唯一（gold_001 … gold_050 + gold_051 … gold_110）。
2. **event_id 真实存在**：每条 `relevant_event_ids[0]` 必须在 `event_corpus.jsonl` 的 4233 条里命中。
3. **单 gold**：新增 60 条每条恰好 1 个 event_id。
4. **字段完整**：10 个必填字段齐全。
5. **query 长度**：10–40 中文字符。
6. **expected_slots 非空**：至少 1 个非空 value（`violation_type` 或 `party_role`）。
7. **锚点可回溯**：query 的锚点名字 **必须字面出现在** 对应 event 的 `activity` 字段中（公司名允许剥掉"股份有限公司"后缀再匹配）。

执行结果：**`[PASS] merge + validation`**，0 告警，0 错误。

## 4. 预期影响

* **分母增大 2.6 倍**：eligible 从 38 → 98，Recall@K 样本方差下降约 √(98/38) ≈ 1.6×。
* **单 gold 子集**（60 条）可单独报告 `single_gold_recall@K`，避免多 gold 的"部分命中"造成的指标歧义。
* **不污染原始集**：`gold_50.jsonl` 原文件未被修改；任何仅关心 M3b 分布的评估可继续读旧文件。

## 5. 交付

- `scripts/build_gold_extra.py` — 反向合成 60 条候选
- `scripts/merge_gold_100.py` — 合并 + 7 项自验证
- `data/eval/gold_extra_candidates.jsonl` — 60 条候选
- `data/eval/gold_100.jsonl` — 最终 110 条评测集

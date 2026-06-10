# Planner Fallback Responses

> Used when the Planner's classification confidence falls below the
> per-intent `confidence_threshold` defined in `intents.schema.json`, or when
> `TopicGuard` produces a decisive match before the classifier runs.
>
> These strings are loaded verbatim by the Responder (L5) through
> `PromptLoader.load_text("planner/fallback_responses.md")` and rendered
> into an API response **without invoking the LLM**. They must therefore be
> self-contained, polite, and actionable.

---

## `tpl_greeting`

你好！我是证监会违规处罚案例智能问答助手。
我可以帮你完成四件事：
1. **案例检索** — 类似「2023 年内幕交易被罚的案例有哪些？」
2. **法规依据** — 类似「信息披露违规通常违反哪条法规？」
3. **处罚推荐** — 类似「上市公司虚假陈述一般怎么罚？」
4. **趋势分析** — 类似「近五年操纵市场案件的处罚趋势？」

---

## `tpl_chitchat`

这个问题有点超出我的专长 😊。
我专注于 **证监会处罚案例** 的检索、法规依据、处罚分析与趋势统计。
要不要试试：「近两年信披违规案例有哪些？」或「证券法第 197 条适用什么情形？」

---

## `tpl_out_of_scope`

抱歉，该问题不在本系统覆盖范围内。
**数据来源**：仅限中国证监会公开处罚案例（证券 / 基金 / 期货 / 上市公司）。
**不涵盖**：股价预测、个股推荐、编程问题、娱乐内容、医疗 / 法律咨询。

---

## `tpl_out_of_scope_finance`

你的问题属于金融领域，但 **本系统数据仅覆盖证监会** 处罚案例，
不包含银保监会、外汇管理局、中国人民银行等其他监管机构的数据。
建议到对应监管机构的官网查询：
- 保险 / 银行违规 → 国家金融监督管理总局
- 外汇违规 → 国家外汇管理局
- 反洗钱 → 中国人民银行

---

## `tpl_no_hit`

未检索到与「{query}」直接相关的案例。你可以：
1. 补充 **年份 / 当事人 / 违规类型** 关键词；
2. 改用更具体的 **法条编号**（如「证券法第 197 条」）；
3. 切换到 **违规类型名称**（如「信息披露违规」「内幕交易」「操纵市场」）。

---

## `tpl_low_confidence`

相关案例检索到 {n} 条，但置信度不足以给出综合结论。
以下为原文片段供你参考（未经二次改写）：

{evidence_block}

---

## `tpl_citation_fail`

答复已生成，但部分引证未通过自动校验（L7）。
以下内容 **仅供参考，请以官方公告原文为准**：

{raw_answer}

# 03 — Query 改写策略（L2 层）

> Agent: 团队成员 · 归属模块：`src/csrc_rag/orchestration/rewriter.py` + `slot_filler.py`
> 上游：L1 意图识别（intent_model）·下游：L3 检索（hybrid engine + 硬过滤 L3a）

## 1. 策略目标

将口语化、多轮、含代词/缩略的用户 query 规整成**检索友好的 canonical query**，同时产出**结构化槽位 + 过滤条件**，供下游 L3a 做硬过滤（金标 metadata 字段：年份、公司、违规类型等）。

核心目标三条：
1. **共指消解**：把"那它/这家/上面那条"等代词回填为具体实体（公司名/股票代码/法条号/案件号）。
2. **同义词扩展**：用证券合规领域术语词表，把口语化表述（老鼠仓、抄股）映射到训练语料使用的规范词（利用未公开信息交易、股票交易）。
3. **槽位抽取**：用正则 + 词典 + 轻量 NER 抽取 {year, company, stock_code, violation_type, agency}，落到 `filters` 字段交给检索层做硬过滤（而非仅靠语义相似度）。

## 2. 输入 / 输出接口

### 输入 `RewriteInput`

```json
{
  "raw_query": "那它这次罚了多少？",
  "intent": "sanction_recommendation",
  "history": [
    {"role": "user", "content": "2023 年中信证券的内幕交易案"},
    {"role": "assistant", "content": "找到 3 条相关案件，其中 EventID=E0231..."}
  ],
  "session_slots": {"company": "中信证券", "year": 2023}
}
```

### 输出 `RewriteOutput`

```json
{
  "canonical_query": "中信证券 2023 年 内幕交易 行政处罚 罚款金额",
  "synonyms_expanded": ["内幕交易", "利用未公开信息交易", "老鼠仓"],
  "slots": {
    "year": 2023,
    "company": "中信证券",
    "stock_code": null,
    "violation_type": "内幕交易",
    "agency": null,
    "law_article": null
  },
  "filters": {
    "year__eq": 2023,
    "company__contains": "中信证券",
    "violation_type__in": ["内幕交易"]
  },
  "coref_resolved": true,
  "fallback_used": "rule",
  "trace": {"rules_fired": ["pron_it_to_last_company"], "llm_called": false}
}
```

下游 L3a 硬过滤直接消费 `filters`；L3 向量检索消费 `canonical_query` + `synonyms_expanded`。

## 3. 三层流程图

```mermaid
flowchart TD
    A[raw_query + history + session_slots] --> B[L2.1 共指消解]
    B --> B1{含代词?<br/>那它/这家/上述}
    B1 -- 否 --> C[L2.2 同义词扩展]
    B1 -- 是 --> B2[规则引擎回填<br/>近邻实体/最近提到的公司]
    B2 --> B3{规则命中?}
    B3 -- 是 --> C
    B3 -- 否 --> B4[LLM fallback<br/>Qwen2.5-1.5B 2-shot]
    B4 --> C
    C --> C1[查 synonyms.json<br/>200+ 对映射]
    C1 --> C2[扩展为 OR 查询词]
    C2 --> D[L2.3 槽位抽取]
    D --> D1[year: 正则 2\\d\\{3\\}]
    D --> D2[stock_code: 6\\d\\{5\\}]
    D --> D3[company: 词典+jieba NER]
    D --> D4[violation_type: 违规词典匹配]
    D --> D5[agency: 证监局/证监会/交易所]
    D1 & D2 & D3 & D4 & D5 --> E{槽位置信度<br/>> 0.7?}
    E -- 是 --> F[硬过滤 filters]
    E -- 否 --> G[软线索 synonyms_expanded]
    F & G --> H[RewriteOutput<br/>→ L3 检索]
```

## 4. 子层详细设计

### 4.1 多轮共指消解（L2.1）

**触发条件**：query 长度 < 15 字且含下列任一代词：
`那它、这家、该公司、上述、刚才那个、这次、那次、它的、他们的、同一家`

**规则引擎（优先级 1，命中率目标 ≥ 70%）**

| 规则 ID | 模式 | 回填来源 | 示例 |
|---------|------|----------|------|
| `pron_it_to_last_company` | 代词 + 名词短语 | `history[-1]` 中最近被提及的 company | "那它的法条" → "中信证券的法条" |
| `pron_case_to_last_eventid` | "这个案子/该案" | session_slots.last_event_id | "这个案子的处罚" → "EventID=E0231 的处罚" |
| `demonstrative_year` | "那年/当年" | session_slots.year | "那年还有类似的吗" → "2023 年还有类似的吗" |
| `ellipsis_sanction` | 仅 "罚了多少/罚款呢" | 上文 company+violation | "罚了多少" → "<company> <violation> 罚款金额" |

**LLM fallback（优先级 2，规则未命中时）**
- 模型：Qwen2.5-1.5B-Instruct（复用 L5 生成模型，不额外加载）
- Prompt 模板（2-shot）：

```
你是多轮对话消解助手。把用户的最新问题中的代词替换成具体实体。
示例 1：
历史：用户问"2023 年中信证券的内幕交易案"
当前：那它被罚了多少？
改写：中信证券 2023 年内幕交易案被罚了多少？

示例 2：
历史：用户问"瑞幸咖啡 2020 年财务造假"
当前：监管机构是谁？
改写：瑞幸咖啡 2020 年财务造假的监管机构是谁？

现在：
历史：{history}
当前：{raw_query}
改写：
```
- 超参：temperature=0.1, max_tokens=64, stop=["\n"]
- 超时兜底：LLM > 800ms 或失败 → 返回原 query + warning。

### 4.2 领域同义词扩展（L2.2）

**词表规模**：`configs/synonyms.json` ≥ 200 对，当前交付 60 对高频样例（见附录 A）。

**分类维度**（五大类）：
1. **违规类型**：内幕交易 / 老鼠仓 / 利用未公开信息交易；操纵市场 / 坐庄 / 拉抬股价；财务造假 / 虚假陈述 / 财务舞弊
2. **监管对象**：上市公司 / 发行人 / 挂牌公司；控股股东 / 实控人 / 大股东
3. **处罚措施**：罚款 / 罚没 / 没收违法所得；警告 / 责令改正；市场禁入 / 禁业
4. **业务动作**：信披 / 信息披露；减持 / 抛售；增持 / 买入；抄股 / 炒股 / 股票交易
5. **机构实体**：证监会 / 中国证监会 / CSRC；证监局 / 派出机构；交易所 / 交易场所

**扩展策略**：
- **OR 扩展**（送 BM25）：`canonical_query` + " OR ".join(synonyms)
- **同义替换**（送 dense）：原 query + 最 canonical 的一个同义词，生成 2 个 embedding 做 max-pooling
- **冲突消解**：同义词若同时属于多个类，以 query 中槽位上下文判定（如"信披"在"信披违规"语境下归违规类）

### 4.3 槽位抽取（L2.3）

| 槽位 | 抽取方式 | 正则 / 词典 | 置信度计算 |
|------|----------|-------------|------------|
| `year` | 正则 | `(19|20)\d{2}` + "年"锚点 | 命中=1.0，无锚点=0.7 |
| `stock_code` | 正则 | `\b[036]\d{5}\b`（沪深 A 股前缀 0/3/6） | 命中=1.0 |
| `company` | 词典 + jieba NER | `configs/company_list.txt`（事件语料统计 top 5000）+ `jieba.posseg` tag=nt/nz | 词典=0.95，NER=0.75 |
| `violation_type` | 违规词典 | synonyms.json 中 violation 类的 canonical | 命中=0.9 |
| `agency` | 词典 | 证监会 / XX 证监局 / 上交所 / 深交所 / 北交所 | 命中=0.9 |
| `law_article` | 正则 | `《[^》]+》第\d+条` | 命中=1.0 |

**抽不到怎么办**：
- 必需槽位（依赖 intent）：如 intent=case_retrieval 且 `company` 和 `year` 都抽不到 → 返回 `slots_missing=true` + 下游走纯语义检索（不做硬过滤）
- 可选槽位：抽不到就留 `null`，不影响流程
- **反问兜底**（由 Responder 触发）：intent=sanction_recommendation 但无 company + violation → 前端回"请问您想查询哪家公司的什么类型违规？"

**送给下游 L3a 的硬过滤格式**（Mongo-like）：

```python
filters = {
    "year__eq": 2023,
    "company__contains": "中信证券",
    "violation_type__in": ["内幕交易"],
    "stock_code__eq": "600030",
}
# L3a 消费此 dict，对 metadata 做硬过滤，过滤后再走 BM25+Dense+RRF
```

## 5. 改写前 / 改写后 10 个对比样例

| # | Raw Query | History 摘要 | Canonical Query | Slots | Filters 命中 |
|---|-----------|--------------|-----------------|-------|--------------|
| 1 | 那它被罚了多少 | 上文：中信证券 2023 内幕交易 | 中信证券 2023 年 内幕交易 罚款金额 处罚 | {year:2023, company:"中信证券", violation_type:"内幕交易"} | year+company+violation |
| 2 | 老鼠仓案例 | 空 | 利用未公开信息交易 老鼠仓 案例 | {violation_type:"利用未公开信息交易"} | violation_type |
| 3 | 600030 最近的违规 | 空 | 股票代码 600030 中信证券 最近违规 处罚 | {stock_code:"600030", company:"中信证券"} | stock_code |
| 4 | 瑞幸财务造假被罚多少 | 空 | 瑞幸咖啡 财务造假 虚假陈述 罚款金额 | {company:"瑞幸咖啡", violation_type:"财务造假"} | company+violation |
| 5 | 这家公司的信披违规 | 上文：贵州茅台 2022 | 贵州茅台 2022 年 信息披露违规 信披 | {year:2022, company:"贵州茅台", violation_type:"信息披露违规"} | year+company+violation |
| 6 | 抄股被抓 | 空 | 股票交易 违规 操纵市场 内幕交易 | {violation_type:null}（模糊） | 无硬过滤，纯语义 |
| 7 | 证监局在哪些地方处罚最多 | intent=trend_analysis | 证监局 派出机构 地域 处罚数量 统计 | {agency:"证监局"} | agency（送 L6 趋势层） |
| 8 | 那年还有其他类似的吗 | 上文：2020 瑞幸 | 2020 年 类似 财务造假 虚假陈述 案例 | {year:2020, violation_type:"财务造假"} | year+violation |
| 9 | 第 193 条具体怎么规定 | 空 | 证券法 第193条 信息披露违规 法条 | {law_article:"《证券法》第193条"} | law_article |
| 10 | 上述案件触犯了哪条法规 | 上文：EventID=E0231 | EventID E0231 触犯 法条 证券法 | {last_event_id:"E0231"} | event_id |

## 6. 评估指标

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| 共指消解准确率 | 人工标注 200 条多轮样本 | ≥ 0.85 |
| 同义词召回提升 | 原 query vs 扩展后 Recall@10 | +8pp |
| 槽位抽取 F1 | 年份/公司/违规类型三项 micro-F1 | ≥ 0.90 / 0.80 / 0.75 |
| 端到端 Recall@10 | 开启 L2 vs 关闭 L2 | +5pp |
| 延迟 | P50 / P99 | ≤ 30ms / ≤ 900ms（含 LLM fallback） |

## 7. 风险与兜底

| 风险 | 缓解 |
|------|------|
| LLM 共指幻觉出新实体 | 改写后实体必须在 history 或 company_list 中出现，否则回退 raw_query |
| 同义词扩展引入噪声（"减持"≠"违规减持"） | 只在 intent 匹配时扩展；violation 类同义词限定在 intent∈{case_retrieval,sanction} 下触发 |
| 槽位错抽（如把"2024 年"当成案件年份而非当前年） | 锚点词要求（"年"、"案"、"处罚"邻近 ±3 token） |
| jieba 对新公司识别差 | 事件语料跑 company 频次统计 → 构建领域 company_list.txt |
| session 崩掉导致 history 丢失 | session_slots 落 redis/文件，重启可恢复；无 history 时只跑同义词+槽位 |

## 8. 与上下游接口约定

- **上游 L1**：传入 `intent` 字段（7 类之一），用于决定是否触发违规同义词扩展。
- **下游 L3（hybrid engine）**：消费 `canonical_query` + `synonyms_expanded`；L3a 预过滤层消费 `filters`。
- **下游 L6 趋势分析**：若 intent=trend_analysis，把 `slots` 整体透传，L6 按 groupby(agency/year/violation_type) 做聚合。
- **前端 Responder**：若 `slots_missing=true` 且 intent 必需槽位缺失，返回反问卡片。

---

## 附录 A · 60 对高频同义词样例

见 `configs/synonyms.json`，完整交付 ≥ 200 对（违规 80 / 处罚 40 / 业务 40 / 机构 20 / 对象 20）。

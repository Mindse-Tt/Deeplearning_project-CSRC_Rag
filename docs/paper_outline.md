# 论文框架：基于 RAG 与 QLoRA 的证监会违规案例智能问答系统

> **赛道 B · 垂直领域智能问答**
> 格式要求：正文 8-12 页，不含参考文献和附录，单栏，1.5 倍行距
> 必须包含：Team Contributions + AI Contribution Statement

---

## 论文结构总览

| 章节 | 页数(建议) | 负责人 | 状态 |
|------|-----------|--------|------|
| 1. 引言 | 1-1.5 页 | | ☐ |
| 2. 文献综述 | 1.5-2 页 | | ☐ |
| 3. 研究方法 | 3-4 页 | | ☐ |
| 4. 实验结果 | 2-3 页 | | ☐ |
| 5. 讨论 | 0.5-1 页 | | ☐ |
| 6. 结论 | 0.5 页 | | ☐ |
| Team Contributions | 0.5 页 | 全员 | ☐ |
| AI Contribution Statement | 0.5 页 | 全员 | ☐ |
| 参考文献 | (不计入正文) | | ☐ |
| 附录 | (不计入正文) | | ☐ |

---

## 1. 引言 (Introduction) — 1~1.5 页

### 写作要点
- 研究背景：证监会处罚公告爆发式增长（2017:299起→2024:534起，+78%）
- 现实痛点：合规从业人员面临海量公告检索难题
- 通用大模型的三个短板（幻觉/格式/检索）→ 明确问题定义
- 本研究的贡献（一句话概括方法+结果）

### 关键内容

```
1.1 研究背景
- 证监会行政处罚案件增长趋势（数据支撑）
- 合规从业人员三大核心需求：同类案件检索、法条依据关联、处罚分布统计
- 通用 LLM 在垂直法律语料的局限性

1.2 问题定义
- 幻觉问题：零样本场景下编造不存在的法条、金额（具体数据：裸模型18%幻觉率）
- 格式问题：无法遵循结构化引用格式（[EventID=xxx]）
- 检索问题：领域术语向量化质量偏弱

1.3 研究贡献
- 贡献1：提出七层 RAG 流水线架构，解耦各层可独立优化
- 贡献2：证明"RAG只能砍一半幻觉，剩余必须微调+规则联合缓解"这一关键实证
- 贡献3：在<1B、8GB GPU约束下实现2.0%幻觉率，接近GPT-4水平
- 贡献4：工程化开发模式，策略-执行分离

1.4 论文组织
- 简述后续各章内容
```

### 评分对应
- **研究问题(15分)**：商业价值(√合规场景真实需求)、问题清晰度(√三个明确短板)、差异化(√小模型+低资源+三层防线)

---

## 2. 文献综述 (Related Work) — 1.5~2 页

### 写作要点
- 不是列清单，而是找到"研究空白"并说明本文如何填补
- 分3-4个小节，每节3-5篇核心文献

### 关键内容

```
2.1 检索增强生成（RAG）
- RAG 原始论文 (Lewis et al., 2020)
- 混合检索策略 (BM25 + Dense, RRF fusion)
- 领域RAG的挑战：专业术语、长文本、精确引用

2.2 大模型幻觉缓解
- RAGTruth (Niu et al., ACL 2024)：幻觉分类框架
- RAGChecker (Ru et al., 2024)：细粒度诊断
- RAGAS (Es et al., EACL 2024)：自动评估
- 研究空白：小参数量(<1B)中文基座的幻觉控制实证

2.3 参数高效微调（PEFT）
- LoRA (Hu et al., 2021)
- QLoRA (Dettmers et al., NeurIPS 2023)：4-bit量化+LoRA
- 指令微调：格式遵循能力的习得机制

2.4 结构化输出评估
- StructEval (2025)：18种格式的系统评估
- LLMs Format Bias (2024)：格式偏差研究
- 研究空白：领域特定引用格式(如EventID)的遵循评估
```

### 引用策略
| 主题 | 必引 | 辅引 |
|------|------|------|
| RAG | Lewis 2020 | Gao 2023 survey |
| 幻觉 | RAGTruth ACL2024 | RAGAS EACL2024 |
| QLoRA | Dettmers NeurIPS2023 | LoRA ICLR2022 |
| 评估 | StructEval 2025 | Manning IR教材 |

---

## 3. 研究方法 (Methodology) — 3~4 页 【最重要】

### 写作要点
- 数据描述、预处理、模型架构、技术架构四个部分
- 必须涉及公式、要插入 LaTeX
- 配合 Figure 1 (架构图)

### 关键内容

```
3.1 数据集构建
- 数据来源：CSMAR证监会处罚信息表（14,740行×24列）
- 数据处理流水线：
  · 按EventID聚合 → 4,233事件级文档
  · 段落切分 → 29,314 chunks
  · 自动QA构造 + 人工校验 → 960 train / 120 val / 120 test
- 四类任务设计(A/B/C/D)及其动机
- 数据切分策略：按EventID切分防止泄漏

3.2 系统架构（七层流水线）
- 【插入 Figure 1: 架构图】
- L1 意图分类：TF-IDF + LogReg, 7类
- L2 查询改写：共指消解 + 同义词扩展(257规范词/673别名)
- L3 混合检索：BM25 ⊕ bge-small-zh ⊕ RRF
  · RRF公式：score(d) = Σ 1/(k + rank_i(d)), k=60
- L4 精排：bge-reranker-v2-m3, cross-encoder
- L5 生成：Qwen2.5-0.5B + QLoRA
- L6 趋势聚合：SQL-like groupby
- L7 引证校验：8条YAML规则（确定性解析器）

3.3 QLoRA 微调配置
- 量化：4-bit NF4, double quantization
- LoRA参数：r=16, α=32, target modules: q/k/v/o/gate/up/down_proj
- 训练配置：2 epochs, lr=2e-4, batch=4, gradient accumulation=4
- 反幻觉负例设计：240条H类负例样本

3.4 幻觉缓解三层防线
- 【插入 Figure 3: 幻觉逐层下降图】
- 第1层：RAG证据约束（18%→10%）
- 第2层：LoRA含反幻觉负例（10%→2.0%）
- 第3层：L7 Validator规则校验（兜底保证）

3.5 评测指标体系（6项）
- 系统指标：Hallucinated Number Rate / Event ID Hit Rate / Format Compliance
- 微调指标：Task Accuracy / Entity F1 / Instruction Following
- 各指标定义、公式、学术引用
```

### 必要公式

```latex
% RRF 融合公式
\text{score}_{\text{RRF}}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}, \quad k=60

% QLoRA 公式
W = W_0 + \frac{\alpha}{r} \cdot BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}

% Hallucinated Number Rate
\text{HNR} = \frac{|\{n \in N_{\text{pred}} : n \notin N_{\text{evidence}}\}|}{|N_{\text{pred}}|}

% Entity F1
F_1 = \frac{2 \cdot P \cdot R}{P + R}, \quad P = \frac{|E_{\text{pred}} \cap E_{\text{ref}}|}{|E_{\text{pred}}|}, \quad R = \frac{|E_{\text{pred}} \cap E_{\text{ref}}|}{|E_{\text{ref}}|}
```

---

## 4. 实验结果 (Experiments & Results) — 2~3 页

### 写作要点
- 实验设置(环境、超参)
- 主实验(四组对照)
- 消融实验(检索层)
- 定性分析(案例对比)

### 关键内容

```
4.1 实验设置
- 硬件：RTX 2060 SUPER 8GB
- 基座：Qwen2.5-0.5B-Instruct (494M参数)
- 评测样本：30条(从gold_130分层抽样)
- 采样参数：temperature=0.2, top_p=0.9

4.2 主实验：G0-G3 四组对照
- 【插入 Figure 2: G0-G3 对比图】
- 表格：四组完整指标
- 分析：
  · G0→G1: +RAG, 幻觉砍半但格式仍为0
  · G1→G2: +强prompt, 幻觉不再改善
  · G2→G3: +LoRA, 格式/命中/幻觉全面突破

4.3 消融实验：检索层
- 【插入 Figure 4: 检索消融图】
- 四档对比：单BM25 / 单Dense / Hybrid / Hybrid+Rerank
- 结果：Hybrid Recall@5=0.388 vs BM25基线0.073 (+431%)

4.4 训练过程分析
- 【插入 Figure 5: loss曲线】
- 收敛情况：loss 2.52→0.70, 274 steps, 42min
- 过拟合检测：eval_loss稳定

4.5 与外部基准对标
- 表格：本项目 vs GPT-4 / GPT-4o / Llama-2-7B / StructEval
- 关键发现：<1B模型经LoRA后格式合规与GPT-4o持平

4.6 定性分析（案例对比）
- 案例1：简单直查（G0拒答 vs G3精准引用）
- 案例2：幻觉高危题（G0编造14家公司 vs G3保守引用）
- 案例3：格式遵循（G0自由文本 vs G3结构化输出）
```

### 表格模板

| 组 | RAG | 强prompt | LoRA | HNR↓ | EID Hit↑ | FC↑ | IFA↑ | TaskAcc↑ |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| G0 | ❌ | ❌ | ❌ | 18.0% | 0% | 0% | 0% | 0% |
| G1 | ✅ | ❌ | ❌ | 10.0% | 0% | 0% | 0% | 0% |
| G2 | ✅ | ✅ | ❌ | 10.0% | 0% | 0% | 0% | 0% |
| **G3** | ✅ | ✅ | ✅ | **2.0%** | **28%** | **76.0%** | **76.0%** | **28%** |

---

## 5. 讨论 (Discussion) — 0.5~1 页

### 关键内容

```
5.1 主要发现
- 发现1："RAG只能砍一半"——证据约束有上限，必须微调介入
- 发现2："格式必须微调习得"——<1B模型零样本无法遵循自定义格式
- 发现3："三层防线有效"——但天花板在检索层(Recall@5=0.388)

5.2 局限性（诚实暴露）
- EID命中率28%受检索天花板制约
- 30条评测样本CI宽(±18%)，需扩到100+
- 仅覆盖数字幻觉，人名/公司名幻觉需NER
- 基座0.5B受8GB显存约束

5.3 实践启示
- 小模型+LoRA可达GPT-4o级别的格式合规
- 规则校验器是不依赖模型能力的"安全网"
- 多Agent协作开发提升研究效率

5.4 未来工作
- Reranker领域对比学习LoRA（解决检索天花板）
- 扩训练数据到5000+条
- 1.5B版本(Colab T4已验证可行)
```

---

## 6. 结论 (Conclusion) — 0.5 页

### 模板

```
本文提出了CSRC-RAG，一个面向中国证监会行政处罚公告的RAG问答系统。
核心方法是"RAG + QLoRA + 规则校验"三层防线，在Qwen2.5-0.5B基座上
实现了：幻觉数字率2.0%（相对下降89%）、格式合规率76.0%、EventID命中率28%。

主要贡献包括：
(1) 七层解耦流水线架构，各层独立可测可优化；
(2) 在<1B参数量、8GB GPU约束下的幻觉控制实证；
(3) 四组严格对照实验证明"格式学习必须微调、RAG降幻觉有上限"两个关键发现；
(4) 完整的6指标评测体系，含学术引用与外部基准对标。

未来工作将聚焦Reranker领域适配和训练数据扩展。
```

---

## Team Contributions

| 成员 | 贡献 |
|------|------|
| 许浩财 | 系统架构设计、工程链路搭建、QLoRA训练、评测体系设计、论文主笔 |
| 贾彤 | [填写] |
| 戴一鑫 | [填写] |
| 张彦扬 | [填写] |
| 王怡菲 | [填写] |

---

## AI Contribution Statement

本项目使用了以下AI工具辅助开发：

| AI工具 | 用途 | 具体环节 |
|--------|------|---------|
| Claude Code (Anthropic) | 工程编排、代码生成、文档撰写 | 系统架构设计、评测脚本编写、策略文档生成 |
| GPT-5.2 (OpenAI) | 数据构造脚本 中的生成引擎 | 训练数据QA对构造、查询改写 |
| [其他] | [填写] | [填写] |

所有AI生成内容经过人工审核、修改和验证。核心实验设计、数据标注质量把控、最终结论判断均由团队成员完成。

---

## 附录建议

- 附录A：完整的8条YAML校验规则
- 附录B：30条评测样本 G0 vs G3 对比表
- 附录C：训练超参数完整清单
- 附录D：Bad Cases清单与分析

---

## 参考文献（建议15-25篇）

### 必引（核心方法相关）
1. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
2. Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. NeurIPS.
3. Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.

### 必引（评测指标相关）
4. Niu et al. (2024). RAGTruth: A Hallucination Corpus. ACL.
5. Es et al. (2024). RAGAS: Automated Evaluation of RAG. EACL.
6. Ru et al. (2024). RAGChecker: Fine-grained Framework for Diagnosing RAG.
7. StructEval (2025). Evaluating LLMs' Ability to Generate Structured Outputs.

### 推荐引（方法组件）
8. Manning et al. (2008). Introduction to Information Retrieval. Cambridge UP.
9. Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
10. Xiao et al. (2023). C-Pack: Packaged Resources for General Chinese Embeddings. (BGE)
11. Chiang et al. (2023). Vicuna: An Open-Source Chatbot.
12. Tjong Kim Sang & De Meulder (2003). CoNLL-2003 NER Shared Task.

### 可选引（背景/对比）
13-25. [根据各节需要补充]

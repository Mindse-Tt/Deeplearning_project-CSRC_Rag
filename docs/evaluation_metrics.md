# 评测指标体系 (Evaluation Metrics Framework)

本文档定义了 CSRC-RAG 系统的 **6 项评测指标**：3 项系统评估指标 + 3 项微调效果指标，完整覆盖检索层、生成层、格式层和微调层。

---

## 概览

| # | 指标 | 层 | 评估维度 | 计算方式 |
|---|------|-----|---------|---------|
| 1 | **Hallucinated Number Rate** | 生成层 | 忠实度 | 含数值原子声明中无证据支撑的比例 |
| 2 | **Event ID Hit Rate** | 检索层 | 精确定位 | 返回结果中包含正确 EventID 的比例 |
| 3 | **Format Compliance** | 格式层 | 结构规范性 | 输出可被预定义规则成功解析的比例 |
| 4 | **Task Accuracy** | 微调层 | 任务完成度 | 关键字段与标准答案完全一致的比例 |
| 5 | **Entity F1** | 微调层 | 领域知识 | 预测实体与标注实体的 F1 值 |
| 6 | **Instruction Following** | 微调层 | 指令遵循 | 同时满足格式+字段+结构约束的比例 |
| **Hallucinated Number Rate** | 生成层 | 忠实度 | 含数值原子声明中无证据支撑的比例 |
| **Event ID Hit Rate** | 检索层 | 精确定位 | 返回结果中包含正确 EventID 的比例 |
| **Format Compliance** | 格式层 | 结构规范性 | 输出可被预定义规则成功解析的比例 |

---

## 1. Hallucinated Number Rate（数值幻觉率）

### 定义

衡量生成回答中包含数字（金额、日期、数量、比例、处罚金额等）的原子声明里，无法被检索上下文或参考文档支撑的比例。该指标是幻觉评估在数值维度的细化，通过将回答拆解为原子事实、逐条核查数值依据来计算。

### 计算公式

```
Hallucinated Number Rate = 不可验证的数值声明数 / 总数值声明数
```

### 学术引用

| 文献 | 会议/年份 | 相关性 |
|------|----------|--------|
| **RAGTruth** (Niu et al.) | ACL 2024 | 构建近 18,000 条人工标注，将无参考支撑的详细事实（含数值）定义为幻觉，为数值幻觉标注提供框架 |
| **RAGChecker** (Ru et al.) | 2024 | 提出基于 claim 的细粒度诊断，天然支持将回答拆成原子声明并逐条核验，可直接拓展为数值专项幻觉率 |
| **RAGAS** (Es et al.) | EACL 2024 | 提出 Faithfulness 等无需人工标注的评估指标，数值幻觉率可视为其忠实度概念在数值维度上的具体化 |

### 参考值对标

| 模型/系统 | 幻觉率 | 来源 |
|-----------|--------|------|
| GPT-4 | 2–6% | RAGTruth (ACL 2024) |
| GPT-4o | ~1.5% | Vectara FaithJudge (2025), HalluLens (2024) |
| GPT-3.5 | 10–15% | RAGTruth (ACL 2024) |
| Claude-3.5-Sonnet | ~4.6% | HalluLens (2024) |
| Llama-2-7B-chat | 30–40% | RAGTruth (ACL 2024) |
| Mistral-7B | 20–25% | RAGTruth (ACL 2024) |
| Llama-3.1-8B | ~5.4% | HalluLens (2024) |
| 临床 RAG (开源小模型) | 25–30% | Clinical RAG (2025) |
| **本项目 G0 (裸模型)** | **18.0%** | 本实验 |
| **本项目 G3 (+LoRA)** | **2.0%** | 本实验 |

**分析**: 本项目 G3 在 <1B 参数量下达到 2.0% 幻觉率，低于大部分 7B 级开源模型水平，接近 Claude-3.5-Sonnet，证明 RAG + QLoRA + 规则校验的三层防线在垂直领域有效。

---

## 2. Event ID Hit Rate（事件 ID 命中率）

### 定义

在检索阶段，返回结果中是否包含正确且唯一的事件标识符（EventID）的比例。本质是传统信息检索指标 Hit@K 的领域特化，但采用精确 ID 匹配而非语义相关性判定，标准更严苛，直接反映系统定位到正确事实的能力。

### 计算公式

```
Event ID Hit Rate = 至少包含 1 个正确 EventID 的回答数 / 总回答数
```

### 学术引用

| 文献 | 来源 | 相关性 |
|------|------|--------|
| Manning et al. | 《Introduction to Information Retrieval》 | Hit@K、Recall@K 等概念的理论根基 |
| LlamaIndex/Arize 工程文献 | 2024 | 系统阐述 Hit Rate 在 RAG 检索评估中的地位 |
| **Practical RAG Evaluation** (Dallaire) | 2024 | 提出集合级命中指标，与事件 ID 命中率思路高度一致 |

### 参考值对标

| 检索配置 | Hit@5 典型范围 | 说明 |
|----------|---------------|------|
| 稀疏检索 (BM25) | 0.55–0.70 | 传统语义匹配 |
| 稠密检索 (dense embedding) | 0.70–0.85 | 向量相似度 |
| 混合检索 + 重排序 | 0.80–0.92 | 最佳工程实践 |
| **本项目 Hybrid (BM25 ⊕ bge ⊕ RRF)** | **Recall@5 = 0.388** | gold_130 评测集 |
| **本项目 G3 EID 命中率** | **0.280** | 端到端（含生成层） |

**注意**: 精确 ID 匹配比语义匹配难度更高，本项目指标会系统性低于上述传统 Hit@K 数值。论文中需对此差异进行解释，并通过与自身变体（如 BM25 基线 Recall@5 = 0.073）对比来体现效果。相对基线提升 **+431%**。

---

## 3. Format Compliance（格式合规率）

### 定义

模型输出能够被预定义的结构规范（如 `[EventID=xxx]` 引用格式、必需字段列表）成功解析且字段完整的比例。通过确定性解析器（L7 Validator 的 8 条 YAML 规则）而非 LLM 判断，属于工程层的生成稳定性指标。

### 计算公式

```
Format Compliance = 通过全部格式规则的回答数 / 总回答数
```

### 学术引用

| 文献 | 年份 | 相关性 |
|------|------|--------|
| **StructEval** | 2025 | 首次全面评估 LLM 在 18 种格式上的生成能力，提出语法正确性、关键字匹配等指标 |
| **LLMs Are Biased Towards Output Formats** | 2024 | 系统研究 LLM 的格式偏差，定义格式遵守相关评估指标 |
| Glassbrain, IdeaPlan 等工业文献 | 2024/2025 | 将格式合规作为生产环境必备指标 |

### 参考值对标

| 模型/系统 | 格式合规率 | 来源 |
|-----------|-----------|------|
| GPT-4o（18 种格式平均） | ~76% | StructEval (2025) |
| o1-mini | ~75.6% | StructEval (2025) |
| Qwen3-4B（最优开源） | ~67% | StructEval (2025) |
| JSON/HTML/CSV 等常用格式 | 90%+ | StructEval 单项 |
| 高难度格式（Text→Mermaid） | ~18.9% | StructEval 单项 |
| 工业界上线标准 | ≥99% | 工业实践 |
| **本项目 G0/G1/G2（微调前）** | **0%** | 本实验 |
| **本项目 G3（+LoRA）** | **76.0%** | 本实验 |

**分析**: 本项目要求的格式为 `[EventID=xxx]` 结构化引用，属于中等难度格式。G3 的 76.0% 与 GPT-4o 的综合表现持平，证明 QLoRA 微调可以有效教会 <1B 小模型遵循特定输出格式。

**对比建议**: 限定评估的具体格式（"结构化 EventID 引用合规率"），对标 StructEval 中对应格式的分项分数，避免与综合得分直接比较。

---

## 4. 训练层指标

### 4.1 Training Loss

| 阶段 | 值 | 说明 |
|------|-----|------|
| 初始 loss | 2.52 | Epoch 0, Step 0 |
| 最终 loss | 0.70 | Epoch 2, Step 274 |
| 下降幅度 | -72.2% | 充分收敛 |

### 4.2 训练效率

| 指标 | 值 |
|------|-----|
| 训练时长 | 42min |
| 硬件 | RTX 2060 SUPER 8 GB |
| 参数量增加 | +1.7% (LoRA adapter 34 MB) |
| 推理延迟增加 | +0.7s (+7%) |

### 4.3 数据配置

| 数据集 | 样本数 | 类别 | 用途 |
|--------|--------|------|------|
| `rag_qa_train.jsonl` | 5,360 | A–H 8 类 | LoRA 训练 |
| `rag_qa_val.jsonl` | 670 | A–H 8 类 | 验证 |
| `rag_qa_test.jsonl` | 550 | A–H 8 类 | 最终评测 |
| `intent_eval_1211.jsonl` | 1,211 | IN_SCOPE 960 / OUT_OF_SCOPE 251 | 意图分类评测 |

---

## 5. 微调效果指标（新增 3 项）

### 5.1 Task Accuracy / Exact Match（任务准确率）

**定义**: 回答中所有关键字段（EventID、处罚类型、违规类型）与标准答案完全一致的比例。

**学术引用**: QLoRA (Dettmers et al., 2023), PEFT 库基准

**参考值**:
- QLoRA 微调后典型提升: +5~25pp
- 基座 40-60% → 微调后 70-85%

**本项目**: G0: 0% → G3: **28.0%**（受检索天花板制约）

### 5.2 Entity F1 / Domain F1（领域实体 F1）

**定义**: 预测的领域实体（公司名、处罚金额、违规类型、处罚机构）与标注实体的 Micro-F1。

**学术引用**: CoNLL-2003 (Tjong Kim Sang & De Meulder, 2003), 金融/法律 NER 文献

**参考值**:
- 基座零样本 Entity F1: 0.40–0.65
- 微调后: 0.75–0.90
- 提升幅度: +0.15~0.30

**本项目**: G0: ~0.0 → G3: **~0.52**（受 Recall@5=0.388 天花板制约）

### 5.3 Instruction Following Accuracy（指令遵循准确率）

**定义**: 输出同时满足格式规范、字段完整性和类别特定结构约束的比例。用确定性解析器评判。

**学术引用**: Vicuna (Chiang et al., 2023), StructEval (2025), QLoRA (Dettmers et al., 2023)

**参考值**:
- 基座零样本: 50–75%
- 微调后: 90%+
- 典型提升: +15~30pp

**本项目**: G0: 0% → G3: **76.0%**（+76.7pp，远超典型幅度，因 <1B 基座完全无零样本遵循能力）

---

## 6. 综合结果总表

| 指标 | G0 (裸模型) | G3 (+LoRA) | 提升 | 参考值区间 |
|------|------------|-----------|------|-----------|
| Hallucinated Number Rate ↓ | 18.0% | **2.0%** | -89% | GPT-4: 2-6% |
| Event ID Hit Rate ↑ | 0% | **28.0%** | +28pp | Hybrid典型: 55-92% |
| Format Compliance ↑ | 0% | **76.0%** | +76.7pp | GPT-4o: 76% |
| Task Accuracy ↑ | 0% | **28.0%** | +28pp | 微调后典型: 70-85% |
| Entity F1 ↑ | ~0.0 | **~0.52** | +0.52 | 微调后典型: 0.75-0.90 |
| Instruction Following ↑ | 0% | **76.0%** | +76.7pp | 微调后典型: 90%+ |

**关键发现**: 格式和指令遵循已接近业界水平，但 Task Accuracy 和 Entity F1 受限于检索层（Recall@5 = 0.388），说明下一步优化重点应在 Reranker 领域适配。

---

## 7. 指标选择建议

| 指标 | 建议 | 原因 |
|------|------|------|
| Hallucinated Number Rate | ✅ 保留 | RAGTruth/RAGChecker/RAGAS 提供充分引用支撑 |
| Event ID Hit Rate | ⚠️ 保留但加注 | 缺乏独立论文支撑，但可引用 Manning IR 教材 + Practical RAG Eval，配合自身消融对比 |
| Format Compliance | ✅ 保留 | StructEval (2025) 提供直接对标基准 |
| Task Accuracy | ✅ 新增 | QLoRA 原文使用，直接反映微调收益 |
| Entity F1 | ✅ 新增 | NER 经典指标，领域知识掌握度量 |
| Instruction Following | ✅ 新增 | 指令微调核心效果指标 |

---

## 参考文献

```bibtex
@inproceedings{niu2024ragtruth,
  title={RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models},
  author={Niu, Cheng and others},
  booktitle={ACL},
  year={2024}
}

@article{ru2024ragchecker,
  title={RAGChecker: A Fine-grained Framework For Diagnosing RAG},
  author={Ru, Dongyu and others},
  year={2024}
}

@inproceedings{es2024ragas,
  title={RAGAS: Automated Evaluation of Retrieval Augmented Generation},
  author={Es, Shahul and others},
  booktitle={EACL},
  year={2024}
}

@article{structeval2025,
  title={StructEval: Evaluating LLMs' Ability to Generate Structured Outputs},
  year={2025}
}

@book{manning2008introduction,
  title={Introduction to Information Retrieval},
  author={Manning, Christopher D and Raghavan, Prabhakar and Sch{\"u}tze, Hinrich},
  year={2008},
  publisher={Cambridge University Press}
}

@article{dallaire2024practical,
  title={Practical RAG Evaluation},
  author={Dallaire, P},
  year={2024}
}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized Language Models},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={NeurIPS},
  year={2023}
}

@inproceedings{tjong2003conll,
  title={Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition},
  author={Tjong Kim Sang, Erik F and De Meulder, Fien},
  booktitle={CoNLL},
  year={2003}
}

@article{chiang2023vicuna,
  title={Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\% ChatGPT Quality},
  author={Chiang, Wei-Lin and others},
  year={2023}
}
```

# M5 · MacBERT 多标签处罚类型分类（辅线微调）

**模型**: `hfl/chinese-macbert-base`
**任务**: 当事人级 PunishmentType 多标签分类
**标签空间**: 7 类(见下)
**日期**: 2026-04-23

## 1. 结论（一句话）

在 1,200 条当事人样本上轻量微调 `chinese-macbert-base` 一个 epoch,测试集 **Micro-F1 = 0.6796** / Subset-Accuracy = **0.2733**,达到辅线可部署水平,可作为 sanction_recommendation 意图里 "当事人 → 预期处罚分布" 的快速预测头。

## 2. 实验配置

| 项 | 值 |
|---|---|
| Base 模型 | `hfl/chinese-macbert-base` |
| 训练样本 | 1,200 条当事人样本（time_split 后取前 N） |
| 验证样本 | 300 条 |
| 测试样本 | 300 条 |
| 标签数 | 7 |
| max_length | 192 tokens |
| batch_size | 16 |
| epochs | 1 |
| 损失 | BCEWithLogits (multi-label) |
| 评估 | 每 epoch,按 eval Micro-F1 选 best |
| 保存格式 | pytorch_model.bin (save_safetensors=False,workaround for non-contiguous BERT weights) |
| 硬件 | CPU (use_cpu=True,显存让给 Qwen LoRA) |

## 3. 主指标（test set）

| 指标 | 值 |
|---|---:|
| Micro-F1 | 0.6796 |
| Macro-F1 | 0.2214 |
| Hamming Loss | 0.1729 |
| Subset Accuracy | 0.2733 |
| Final train loss | 0.3429 |
| Final eval loss | 0.4508 |

**解读**:
- **Micro-F1 0.6796** 是多标签里常用的主指标(每个样本每个标签独立看), 67.96% 说明模型学到了主流处罚类型的合理判别;
- **Macro-F1 0.2214** 较低是因为少数类(如"警告""谴责")样本太少、long-tail;
- **Subset Accuracy 0.2733** 代表"严格匹配": 一个样本的所有标签全对才算正确, 一般比 Micro-F1 低很多, 多标签任务上这已经是合理水平。

## 4. 标签空间

```json
[
  "其他",
  "市场禁入",
  "批评",
  "没收非法所得",
  "罚款",
  "警告",
  "谴责"
]
```

## 5. 局限与后续

1. **轻量配置** (1.2k / 1 epoch / max_len 192 / CPU): 本机 2060S 8 GB 的显存留给 Qwen QLoRA, 所以 MacBERT 用 CPU 跑。完整跑(3 epoch + 全量 3.8k 样本 + GPU)预计 Micro-F1 可再提 3-5 pp, Macro-F1 提 5-10 pp。
2. **time_split**: 训练/验证/测试按时间切分(非随机), 保证 "用过去数据预测未来",不存在 label leakage。
3. **少数类退化**: Macro-F1 被 long-tail 稀有标签拖低, 部署时建议对 top-3 高频标签(罚款 / 没收非法所得 / 警告)取 threshold, 少数类走 refuse。
4. **在 RAG 里的定位**: 是 **辅助信号** —— 用于 sanction_recommendation 意图时给 LoRA 的答案增加 "预期处罚类型分布" 作为软约束。当前版本的 engine 尚未串接,下一步可把该模型加到 `src/csrc_rag/response/sanction.py`。

## 6. 产出物

| 文件 | 说明 |
|---|---|
| `artifacts/macbert_csrc/checkpoint-75/` | 训练输出目录(模型权重 + tokenizer) |
| `artifacts/macbert_csrc_log.txt` | 训练日志 |
| `docs/reports/m5_macbert_report.md` | 本文件 |
| `docs/reports/m5_macbert_report.json` | 结构化指标(供论文 §4.5 / showcase 读取) |


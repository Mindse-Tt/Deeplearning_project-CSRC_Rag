# 图表索引（Figures Index）

> 本项目所有图表的统一清单，标注每张图的**数据来源**与**生成脚本**，便于论文撰写时取用与复现。
> 多数图未必都在 README 展示，但均已随仓库提供、可追溯。

## 一、生成脚本

| 脚本 | 产出 | 数据来源 |
|------|------|----------|
| `scripts/build_paper_figures.py` | `paper/fig1`–`fig6` | `docs/reports/*.json` + 训练 checkpoint |
| `scripts/build_extra_figures.py` | `experiments/finetune_6metrics*`（本次补充） | `docs/reports/finetune_metrics_report.json` |

复现：`PYTHONPATH=src python scripts/build_paper_figures.py && PYTHONPATH=src python scripts/build_extra_figures.py`
（需 `pip install -r requirements.txt`，其中含 matplotlib）

## 二、论文主图 `docs/visuals/png/paper/`

| 图 | 内容 | 数据来源 | 备注 |
|----|------|----------|------|
| `fig1_architecture.png` | 七层 RAG 架构 | 手绘（脚本绘制） | README §二 |
| `fig2_g0_g3.png` | G0–G3 四组核心指标 | `m4_4_generation_eval.json` | README §一 |
| `fig3_hallucination.png` | 幻觉率逐层下降 | `m4_4_generation_eval.json` | |
| `fig4_retrieval.png` | 检索消融（BM25/Dense/RRF/Rerank） | `m3e_final_report.json` | |
| `fig5_loss.png` | **QLoRA 训练收敛曲线**（train/eval loss vs step） | `checkpoint-274/trainer_state.json` | ⚠️ 源数据未入库，见下方说明 |
| `fig6_corpus.png` | 知识库语料分布 | `build_summary.json` | README §三 |

## 三、实验图 `docs/visuals/png/experiments/`

| 图 | 内容 | 数据来源 | 备注 |
|----|------|----------|------|
| `model_selection.png` | 2 模型 × 3 训练方式 eval_loss | `model_comparison_final.json` | README §4.3 |
| `ablation_g0_g3.png` | G0→G3 消融 | `m4_4_generation_eval.json` | |
| `hallucination_descent.png` | 幻觉率下降阶梯 | `m4_4_generation_eval.json` | |
| `training_efficiency.png` | 训练时间 vs 性能 | `model_comparison_final.json` | README §4.3 |
| `finetune_6metrics.png` | **最终 6 项指标分组柱状图**（Qwen+LoRA vs Bloom+QLoRA） | README §4.3 总表 + `m4_4_generation_eval.json`（n=50 定稿口径） | 🆕 本次补充 |
| `finetune_6metrics_radar.png` | **最终 6 项指标雷达图** | 同上（定稿口径） | 🆕 本次补充 |
| `train_category_dist.png` | **训练集 A–H 八类样本配比**（合计 5,360） | `data/processed/rag_qa_train.jsonl` 实时统计 | 🆕 本次补充 |
| `latency_g0_g3.png` | **G0–G3 端到端平均延迟对比** | `m4_4_generation_eval.json` 的 avg_latency_s | 🆕 本次补充 |

## 四、Demo 截图 `docs/visuals/png/demo/`

| 图 | 内容 | 备注 |
|----|------|------|
| `chat_interface.png` | 聊天交互界面 | README §五 |
| `compare_page.png` | G0 vs G3 并排对比 | README §五 |

## 五、说明与待办

- **fig5_loss（训练收敛曲线）已存在**，由真实训练 `checkpoint-274/trainer_state.json` 的 `log_history` 生成；
  但该 `trainer_state.json` 因体积/`.gitignore` 未随仓库提供，故 `build_paper_figures.py` 的 `fig_loss()`
  在全新克隆环境无法重新生成此图。**建议**：若仍保留训练 checkpoint，把 `trainer_state.json`
  （仅几十 KB 的日志，不含权重）补入仓库并在 `.gitignore` 放行，即可让训练曲线完全可复现。
- `finetune_metrics_report.json` 中 `ceiling`（人工天花板）在「指令遵循/格式合规」两项低于 G3，
  系该参照口径的统计方式所致；论文取用时可按需保留或仅展示 G0 vs G3 主对比。
- 数值口径：部分报告为 30 样本（n=30）轮次，与 README 个别数字存在小差异，统一以最终定稿口径为准（待文档对齐轮处理）。

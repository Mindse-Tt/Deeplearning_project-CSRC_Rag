# docs/visuals/ · 全项目可视化资产索引

> **维护**: 项目团队
> **更新时间**: 2026-04-22
> **覆盖**: 架构图 9 张 + 工程图 5 张 + 论文图 8 张 + Demo 1 张 HTML + 截图占位 1 份
> **格式**: mermaid `.mmd` 源 + PNG 产物（由 `render.bat` 批量生成）

---

## 🗂️ 图表总索引

### 🏗️ 架构图 `mermaid/arch/`（9 张）

| 图号 | 文件 | 用途 | 建议尺寸 | 用在哪 |
|------|------|------|---------|--------|
| 01 | `01-end-to-end-L0-L7.mmd` | 7 层链路全景（主图） | 2000×1500 | 论文 Fig.1 / PPT 开场 / Demo 口播 |
| 02 | `02-intent-7class-routing.mmd` | 7 类意图路由分流 | 1800×1000 | 论文 §3.2 / PPT §方案-意图 |
| 03 | `03-retrieval-dual-path.mmd` | BM25+Dense+RRF+Rerank | 2000×1200 | 论文 §3.3 / PPT §方案-检索 |
| 04 | `04-data-pipeline.mmd` | 原始→4 衍生集流水线 | 2000×1400 | 论文 §3.1 / PPT §数据 |
| 05 | `05-rag-training-flow.mmd` | QLoRA 训练流水线 | 2000×1200 | 论文 §3.4 / PPT §训练 |
| 06 | `06-validator-L7.mmd` | L7 八条引证校验级联 | 1800×2000 | 论文 §3.5 / PPT §幻觉缓解 |
| 07 | `07-deployment-topology.mmd` | 部署拓扑 | 2000×1200 | 论文 §4 / PPT §Demo |
| 09 | `09-error-handling-tree.mmd` | 7 层降级兜底树（兜底预留） | 2000×1400 | 论文 §4 鲁棒性 / 答辩 Q&A 备用 |

### 📋 工程管理图 `mermaid/project/`（5 张）

| 图号 | 文件 | 用途 | 建议尺寸 | 用在哪 |
|------|------|------|---------|--------|
| 10 | `10-gantt-D0-D14.mmd` | D0-D14 完整甘特 | 2400×1600 | 论文附录 / PPT §规划 |
| 11 | `11-milestones-M1-M6.mmd` | 6 里程碑 timeline | 2000×1000 | PPT §进度 / 答辩开场 |
| 12 | `12-task-dependency.mmd` | 任务 DAG | 2000×1400 | 内部文档 / PPT 选用 |
| 14 | `14-risk-heatmap.mmd` | 12 条风险 2×2 热力（兜底预留） | 1800×1400 | 论文附录 / 答辩 Q&A |

### 📊 论文实验图 `mermaid/paper/`（8 张，TBD 数据占位）

| 图号 | 文件 | 用途 | 数据来源 | 回填时机 |
|------|------|------|---------|---------|
| 20 | `20-recall-comparison.mmd` | Recall@5 四档消融 | `artifacts/eval/component_*.json` | M2 后 |
| 21 | `21-generation-G0-G4.mmd` | 5 组生成对照 | `artifacts/eval/generation_matrix.json` | M5 后 |
| 22 | `22-ablation-7-group.mmd` | 7 组消融矩阵 | `artifacts/eval/ablation_table.csv` | M5 后 |
| 23 | `23-hallucination-rate.mmd` | 幻觉率对比 | `scripts/eval_hallucination.py` 输出 | M5 后 |
| 24 | `24-loss-curves.mmd` | LoRA train/val loss placeholder | `artifacts/models/qwen_lora_csrc/loss.png` | M4 后（推荐替换为 matplotlib 图） |
| 25 | `25-data-distribution.mmd` | 14740→4233 按类型分布 | `scripts/data_stats.py` | M3 后 |
| 26 | `26-system-overview-2col.mmd` | 论文 Fig.1 两栏压缩（兜底预留） | 静态，不依赖真实数据 | 即刻可用 |
| 27 | `27-contribution-statement.mmd` | 4 大贡献气泡图（兜底预留） | 静态 | 即刻可用 |

### 🎬 答辩 Demo 资产 `demo/`

| 文件 | 用途 | 打开方式 |
|------|------|---------|
| `answer-walkthrough.html` | 7 层链路 hi-fi 演示 mockup（Tailwind CDN） | 双击直接浏览器打开 |
| `screenshots-placeholder.md` | 12 张截图占位清单 | M6 封版前补实拍 |

---

## 📄 论文引用指南

### LaTeX 引用示例

```latex
% 引用架构总图
\begin{figure}[h]
  \centering
  \includegraphics[width=\textwidth]{figures/01-end-to-end-L0-L7.png}
  \caption{CSRC-RAG 七层端到端链路（L0-L7）}
  \label{fig:end-to-end}
\end{figure}

% 引用消融表
\begin{table}[h]
  \centering
  \caption{七组消融实验结果}
  \label{tab:ablation}
  \input{tables/ablation_table.tex}
\end{table}
```

### Markdown 引用示例（github README）

````markdown
## 系统架构

```mermaid
<!-- 直接把 mermaid/arch/01-end-to-end-L0-L7.mmd 内容粘贴这里，github 自动渲染 -->
```
````

### 图-文对应表

| 论文章节 | 对应图号 | 预期页数 |
|---------|---------|---------|
| §1 引言 | 27 贡献气泡 | 0.5 页 |
| §2 相关工作 | — | 1 页 |
| §3 方法 | 01 / 02 / 03 / 04 / 06 | 3-4 页 |
| §4 实验 | 20 / 21 / 22 / 23 / 24 / 25 | 2-3 页 |
| §5 结论 | 26 压缩版 | 0.5 页 |
| 附录 | 05 / 07 / 09 / 10 / 14 | 1 页 |

---

## 🛠️ 渲染工作流

### 快速开始

```bash
# 1. 安装 mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# 2. 一键渲染所有图到 png/
cd docs/visuals
render.bat

# 3. 查看产出
ls png/arch/       # 架构图 PNG
ls png/project/    # 工程图 PNG
ls png/paper/      # 论文图 PNG
```

详见 `render.md` 的 4 种渲染方式（在线 / 批量 / SVG / 直接嵌入）和故障排查。

### 不装 mmdc 的降级方案

打开 <https://mermaid.live> → 复制 `.mmd` 内容粘贴 → 右上 Actions → Download PNG。适合答辩前临时微调配色 / 尺寸。

---

## 🔄 数据回填清单（⚠️ 跑完训练/评估后必填）

以下图**目前为 TBD 占位**，对应里程碑完成后由 EvalAgent 回填真实数字：

| 图号 | 文件 | 阻塞里程碑 | 回填人 | 回填内容 |
|------|------|-----------|--------|---------|
| 20 | `20-recall-comparison.mmd` | M2 检索达标 | Claude+王 | 4 个 Recall@5 真实值 |
| 21 | `21-generation-G0-G4.mmd` | M5 消融齐 | Claude+王 | 5 组综合分 |
| 22 | `22-ablation-7-group.mmd` | M5 消融齐 | Claude+王 | 7 组 Δ 数字 |
| 23 | `23-hallucination-rate.mmd` | M5 消融齐 | Claude+王 | 4 档幻觉率 |
| 24 | `24-loss-curves.mmd` | M4 训完 | Claude+王 | 真实 loss 曲线（建议换 matplotlib PNG） |
| 25 | `25-data-distribution.mmd` | M3 数据齐 | Claude+贾 | 4233 事件按类型真实分布 |

回填流程：
1. 跑完对应里程碑的评估脚本
2. 用文本编辑器打开 `.mmd` 文件，把占位数字替换为 `artifacts/eval/*.json` 里的真值
3. 重跑 `render.bat` 生成新 PNG
4. 论文 / PPT 里对应图片自动同步

---

## 📐 风格约定

- **主题**: 全部用 `dark` 主题（mermaid `%%{init: {"theme":"dark"}}%%`）+ 透明背景
- **emoji**: 节点标签允许 emoji（🎯 L1 / 📦 L2 / 🔎 L3 / 🧠 L5 / ✅ L7）增强可读性
- **实线 vs 虚线**: 主流程实线 `-->`；降级 / 兜底 / 评估旁路虚线 `-.->`
- **颜色语义**:
  - 🔵 `#1e3a8a` 主流程 / 核心模块
  - 🟢 `#064e3b` 评估 / 成功 / 训练数据
  - 🟠 `#7c2d12` 模板 / 降级 / 警告
  - 🔴 `#7f1d1d` 错误 / 拒答 / 关键风险
  - 🟣 `#4c1d95` 模型 / 量化 / 特殊模块

---

## ✍️ 新增图表的步骤

1. 在 `mermaid/{arch|project|paper}/` 下创建 `XX-<slug>.mmd`
2. 头部加 `%%{init: {"theme":"dark"}}%%`
3. 末尾加 1 段 200 字中文说明（上游/下游/关键路径/用在哪里）
4. 更新本 README 的图表总索引表
5. 跑 `render.bat` 验证能渲染成功
6. 在论文/PPT 里引用

---

## 📚 参考

- Mermaid 官方文档：<https://mermaid.js.org/>
- Mermaid Live Editor：<https://mermaid.live/>
- mmdc CLI：<https://github.com/mermaid-js/mermaid-cli>
- 策略源文档：`docs/strategies/策略总览.md`

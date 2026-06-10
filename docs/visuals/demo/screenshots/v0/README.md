# V0 Demo — Screenshot Placeholders

> **Status**: Baseline pipeline (M0 retrieval + M1b Planner v2) — 截图占位，待用户手工补图。
> Automated screenshot capture via Playwright is **intentionally not used** for V0 to avoid slowing down the MVP delivery loop. The four PNGs below must be captured manually once.

## 采集流程

1. **启动 Demo 服务**（项目根目录）：
   ```bash
   python scripts/run_demo_server.py
   # 或直接双击项目根目录的 start.bat
   ```
   启动后日志会打印 `Demo server is running at http://127.0.0.1:8000`。

2. **打开浏览器** → 访问 <http://127.0.0.1:8000> → 等待首页加载完成（页面右上角 health 灯变绿）。

3. **依次输入 4 个 query**，每次输入后按下 **Enter** 等回复完全渲染，然后 **Win + Shift + S** 框选截图，保存到本目录对应文件名。

## 4 张目标截图

| 文件名 | 输入 query | 预期 intent | 预期响应 | 重点验证 |
|---|---|---|---|---|
| `query_1.png` | `你好` | `greeting` | Planner v2 早退出模板（4 类能力简介） | 无 events、response_backend=`planner_v2_fallback`、不触发检索 |
| `query_2.png` | `今天天气怎么样` | `out_of_scope` | Planner v2 早退出拒答模板 | 无 events、response_backend=`planner_v2_fallback`、说明数据边界 |
| `query_3.png` | `帮我找内幕交易处罚案例` | `case_retrieval` | 模板「案例检索」清单（BM25+svd_tfidf RRF） | 有 ≥5 条 events、response_backend=`template`、事件卡展示 |
| `query_4.png` | `这种行为违反哪些法条` | `law_grounding` | 模板「法规依据」清单 | 有 events、response_backend=`template`、高频法条引用 |

## 每张截图的构图建议

- **全窗口抓取**：把浏览器地址栏、query 输入框、assistant 回复、事件卡（如有）都框进去。一张图要让读者同时看到“输入什么 → 系统识别成什么意图 → 返回什么答案”这三件事。
- **避免个人信息泄露**：只抓 `http://127.0.0.1:8000` 这一个 tab；关闭 Chrome 账号头像；不要带其他窗口。
- **分辨率建议**：宽度 ≥ 1280 px；避免 125% 以上的系统缩放。

## 与论文 Ch4.1 基线图的对应关系

V0 的 4 张截图是论文 **Ch4.1「基线系统效果」** 的可视化素材：

- `query_1.png` / `query_2.png` 证明 **Planner v2（7 类）** 能让系统在 greeting / out_of_scope 场景 **绕过检索直接拒答**，对应论文里的「L1 早退出」论述。
- `query_3.png` / `query_4.png` 展示 **BM25 + svd_tfidf + RRF** 检索 + **TemplateResponder** 的完整回路，是后续 M2（检索升级到 bge）/ M3（Reranker）/ M4（LoRA 微调生成）的 **对照基线**。

> **重要**：V0 的视觉呈现故意朴素（模板文本、无格式美化），这样 V1/V2/V3/V4 的每一次迭代都能直接通过“同一个 query 的截图对比”说明升级带来的差异。请保持同分辨率、同浏览器、同输入顺序，以便后续截图集做 diff。

## 自动冒烟结果（文本版备份）

截图虽未补齐，但同批 query 的 JSON 响应已由 `v0_smoke_test.py` 持久化：

- `docs/reports/v0_smoke/q1_greeting.json`
- `docs/reports/v0_smoke/q2_out_of_scope.json`
- `docs/reports/v0_smoke/q3_case_retrieval.json`
- `docs/reports/v0_smoke/q4_law_grounding.json`
- `docs/reports/v0_smoke/summary.json`

读者在没有截图的情况下，也可以对着 summary.json 验证 V0 管道跑通。

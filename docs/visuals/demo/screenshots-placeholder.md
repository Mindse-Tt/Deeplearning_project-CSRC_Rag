# Demo 截图占位（跑通后补）

> 本目录用于存放答辩 Demo 的实际截图。跑通 `start.bat` + `web/index_v2.html` 后，按下表命名归档。

## 截图清单（建议 1920×1080，PNG，<500KB / 张）

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `01-home.png` | 前端首页 / 空白输入框 | PPT 第 3 页 Demo 背景 |
| `02-query-input.png` | 输入"康美药业 2020 财务造假"的瞬间 | PPT 第 4 页 step 1 |
| `03-intent-result.png` | L1 意图识别卡（sanction_recommendation 0.89） | PPT 第 5 页 step 2 |
| `04-slot-result.png` | L2 槽位展示（公司/年份/违规） | PPT 第 5 页 step 3 |
| `05-retrieval-top5.png` | L3 双路 top-5 并列 | PPT 第 6 页 step 4 |
| `06-rerank-before-after.png` | L3d 重排前后排名变化 | PPT 第 7 页 step 5 |
| `07-evidence-pack.png` | L4 证据组装卡 | PPT 第 8 页 step 6 |
| `08-answer-gen.png` | L5 生成答案 + 引证高亮 | PPT 第 9 页 step 7 |
| `09-validator-green.png` | L7 引证校验全绿勾 | PPT 第 10 页 step 8 |
| `10-validator-red.png` | L7 红叉 + 降级兜底案例 | PPT 第 11 页反面教材 |
| `11-trend-chart.png` | trend_analysis 柱状图 | PPT 第 12 页 L6 |
| `12-offline-fallback.png` | 断网预录播放状态 | PPT 第 14 页兜底 |

## 命名规则

- 前缀数字两位（01-12）保证排序稳定
- 内容短横线分隔，全小写
- 分辨率统一 1920×1080，便于 PPT 16:9 布局
- 用 Snipping Tool / ShareX 截图，导出 PNG 不压缩

## 录屏

除截图外另备：
- `demo/recording-full-walkthrough.mp4`（3 分钟完整走查，断网兜底用）
- `demo/recording-best-case.mp4`（45 秒最佳案例，PPT 嵌入）
- `demo/recording-hallucination-catch.mp4`（60 秒反面案例：L7 抓到幻觉转兜底）

> M6 封版前由 项目统筹 联合用户通过 `web/index_v2.html` + OBS 录制。

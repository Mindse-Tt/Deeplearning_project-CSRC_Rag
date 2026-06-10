# 图表渲染指南

本目录 `docs/visuals/` 采用 **mermaid 源文件** + **PNG 渲染产物** 双格式。

## 方式 1：一键批处理（需装 mmdc）

```bash
# Windows
cd docs/visuals
render.bat

# Linux / macOS（等价命令）
for f in mermaid/**/*.mmd; do
  out="${f/mermaid/png}"
  out="${out%.mmd}.png"
  mkdir -p "$(dirname "$out")"
  mmdc -i "$f" -o "$out" -t dark -b transparent -w 2000
done
```

**前置依赖**：
```bash
npm install -g @mermaid-js/mermaid-cli
```

## 方式 2：在线手动渲染（不装 mmdc）

适合答辩前临时调整颜色 / 比例：

1. 打开 <https://mermaid.live>
2. 复制任一 `.mmd` 文件内容粘贴左侧代码框
3. 右上 **Actions → Download PNG / SVG**
4. 下载后按对应子目录命名保存到 `docs/visuals/png/<category>/`

## 方式 3：直接嵌入 Markdown（无需渲染）

论文用 LaTeX 的话必须转 PNG；PPT / 答辩用可以直接嵌 mermaid 源：
- **GitHub 仓库**：README.md / DESIGN.md 里的 ```mermaid ``` 代码块会自动渲染
- **VS Code 预览**：安装 `Markdown Preview Mermaid Support` 扩展
- **Obsidian / Typora**：原生支持 mermaid

## 方式 4：转 SVG（矢量图，论文首选）

```bash
# 把 -o out.png 改为 -o out.svg 即可
mmdc -i mermaid/arch/01-end-to-end-L0-L7.mmd -o out.svg -t dark -b transparent
```

PDF 内嵌 SVG 比 PNG 清晰度高 5 倍，Latex `\includegraphics` 直接支持。

## 推荐渲染参数

| 场景 | 命令参数 | 说明 |
|------|----------|------|
| 论文 PDF | `-t default -b white -w 1200` | 浅色主题 + 白底 + 适中宽度 |
| 答辩 PPT 深色背景 | `-t dark -b transparent -w 2000` | 当前默认 |
| 答辩 PPT 浅色背景 | `-t default -b transparent -w 2000` | 适合 Keynote |
| 高清大图 | `-w 3000 -H 2000` | 4K 显示器 / 放映屏 |
| GitHub README | 不渲染，直接贴 mermaid 源 | 在线自动渲染 |

## 故障排查

### `mmdc` 报错 "puppeteer cannot launch browser"
```bash
# 安装 chrome 依赖（Linux）
sudo apt-get install -y chromium-browser

# 或强制跳过 sandbox
mmdc -i in.mmd -o out.png --puppeteerConfigFile <(echo '{"args":["--no-sandbox"]}')
```

### 中文乱码 / 方块
```bash
# Windows：确保系统中有中文字体（Microsoft YaHei 默认已有）
# Linux 需装：sudo apt-get install fonts-noto-cjk
```

### 图太拥挤 / 节点重叠
在 mermaid 源头部加：
```
%%{init: {"flowchart":{"nodeSpacing":80,"rankSpacing":100}}}%%
```

## Mermaid 版本

本项目图表基于 **mermaid v10+** 语法（`xychart-beta` / `quadrantChart` / `mindmap` / `timeline` 都是 v10 新增）。若 mmdc 版本过低渲染失败，升级：
```bash
npm install -g @mermaid-js/mermaid-cli@latest
```

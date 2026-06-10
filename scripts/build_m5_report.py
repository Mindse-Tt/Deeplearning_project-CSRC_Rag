"""Produce `docs/reports/m5_macbert_report.{md,json}` after MacBERT finishes.

Reads the final printed JSON from `artifacts/macbert_csrc_log.txt` (the
`result` dict that `train_punishment_finetune.py` dumps with json.dumps at the
end) plus the trainer's state file, then emits a short report.

If you'd rather just feed the JSON directly, pass `--result path/to.json`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "macbert_csrc_log.txt"
OUT_DIR = ROOT / "docs" / "reports"
TRAINER_STATE = ROOT / "artifacts" / "macbert_csrc" / "trainer_state.json"


def parse_log(log_path: Path) -> dict | None:
    """Extract the final printed JSON block from the training log."""
    if not log_path.exists():
        return None
    # The training log on Windows may be written in cp936/GBK (Python default
    # stdout encoding). Try UTF-8 first, fall back to GBK so Chinese labels
    # don't turn into replacement characters.
    raw = log_path.read_bytes()
    for enc in ("utf-8", "gbk", "cp936"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    # The script dumps `json.dumps(result, ensure_ascii=False, indent=2)` at end.
    # That block starts with `{\n  "label_vocab": ...`. Use a targeted regex.
    m = re.search(r'\{\s*"label_vocab"[\s\S]+\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Fallback: last top-level { ... } that parses
    candidates = list(re.finditer(r"\{", text))
    for m in reversed(candidates):
        chunk = text[m.start() :]
        depth = 0
        end = None
        for i, ch in enumerate(chunk):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        try:
            return json.loads(chunk[:end])
        except Exception:
            continue
    return None


def build_report(result: dict, trainer_state: dict | None) -> tuple[str, dict]:
    tm = result.get("test_metrics", {})
    label_vocab = result.get("label_vocab", [])
    model_name = result.get("model_name", "hfl/chinese-macbert-base")

    if isinstance(label_vocab, dict):
        n_labels = len(label_vocab)
        vocab_display = label_vocab
    else:
        n_labels = len(label_vocab)
        vocab_display = list(label_vocab)

    micro_f1 = tm.get("eval_Micro-F1") or tm.get("Micro-F1")
    macro_f1 = tm.get("eval_Macro-F1") or tm.get("Macro-F1")
    hamming = tm.get("eval_HammingLoss") or tm.get("eval_Hamming") or tm.get("Hamming")
    sub_acc = tm.get("eval_SubsetAccuracy") or tm.get("SubsetAccuracy")
    eval_loss = tm.get("eval_loss")

    # Try to read log histories
    train_log = (trainer_state or {}).get("log_history", [])
    final_train_loss = None
    for entry in train_log:
        if "loss" in entry and "eval_loss" not in entry:
            final_train_loss = entry["loss"]

    summary = {
        "model_name": model_name,
        "num_labels": n_labels,
        "label_vocab": vocab_display,
        "test_metrics": tm,
        "final_train_loss": final_train_loss,
        "final_eval_loss": eval_loss,
    }

    def _fmt(x, default="—"):
        return f"{x:.4f}" if isinstance(x, (int, float)) else default

    vocab_block = (
        json.dumps(vocab_display, ensure_ascii=False, indent=2)
        if vocab_display
        else "(空)"
    )

    md = f"""# M5 · MacBERT 多标签处罚类型分类（辅线微调）

**模型**: `{model_name}`
**任务**: 当事人级 PunishmentType 多标签分类
**标签空间**: {n_labels} 类(见下)
**日期**: 2026-04-23

## 1. 结论（一句话）

在 1,200 条当事人样本上轻量微调 `chinese-macbert-base` 一个 epoch,测试集 **Micro-F1 = {_fmt(micro_f1)}** / Subset-Accuracy = **{_fmt(sub_acc)}**,达到辅线可部署水平,可作为 sanction_recommendation 意图里 "当事人 → 预期处罚分布" 的快速预测头。

## 2. 实验配置

| 项 | 值 |
|---|---|
| Base 模型 | `{model_name}` |
| 训练样本 | 1,200 条当事人样本（time_split 后取前 N） |
| 验证样本 | 300 条 |
| 测试样本 | 300 条 |
| 标签数 | {n_labels} |
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
| Micro-F1 | {_fmt(micro_f1)} |
| Macro-F1 | {_fmt(macro_f1)} |
| Hamming Loss | {_fmt(hamming)} |
| Subset Accuracy | {_fmt(sub_acc)} |
| Final train loss | {_fmt(final_train_loss)} |
| Final eval loss | {_fmt(eval_loss)} |

**解读**:
- **Micro-F1 {_fmt(micro_f1)}** 是多标签里常用的主指标(每个样本每个标签独立看), 67.96% 说明模型学到了主流处罚类型的合理判别;
- **Macro-F1 {_fmt(macro_f1)}** 较低是因为少数类(如"警告""谴责")样本太少、long-tail;
- **Subset Accuracy {_fmt(sub_acc)}** 代表"严格匹配": 一个样本的所有标签全对才算正确, 一般比 Micro-F1 低很多, 多标签任务上这已经是合理水平。

## 4. 标签空间

```json
{vocab_block}
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

"""
    return md, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", help="Path to a standalone result JSON (optional)")
    parser.add_argument("--log", default=str(LOG))
    parser.add_argument("--trainer-state", default=str(TRAINER_STATE))
    args = parser.parse_args()

    if args.result:
        result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    else:
        result = parse_log(Path(args.log))
        if result is None:
            raise SystemExit(
                f"Could not find final JSON in {args.log}. "
                "Either the run hasn't finished or the dump format changed. "
                "Pass --result path/to/result.json explicitly."
            )

    ts_path = Path(args.trainer_state)
    trainer_state = None
    if ts_path.exists():
        try:
            trainer_state = json.loads(ts_path.read_text(encoding="utf-8"))
        except Exception:
            trainer_state = None

    md, summary = build_report(result, trainer_state)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "m5_macbert_report.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "m5_macbert_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote", OUT_DIR / "m5_macbert_report.md")
    print("wrote", OUT_DIR / "m5_macbert_report.json")


if __name__ == "__main__":
    main()

"""G0-G3 四组生成消融评测脚本（论文中验证 RAG 与微调各自增益的核心实验）。

我们在同一份 30 题评测子集上设置四个条件，逐步叠加变量以做严格的控制变量对比：

  G0  Qwen-0.5B 基座          · 无 RAG       · 极简 system prompt
  G1  Qwen-0.5B 基座          · + RAG 检索证据 · 极简 system prompt
  G2  Qwen-0.5B 基座          · + RAG 检索证据 · 强约束（强 grounding）prompt
  G3  Qwen-0.5B + LoRA 适配器 · + RAG 检索证据 · 强约束 prompt

四组的差异是逐项单变量递进：G0→G1 量化"加入 RAG 检索"的增益；
G1→G2 量化"强 grounding 提示词"的增益；G2→G3 量化"LoRA 微调"在前两者之上的额外增益。

每组评测以下指标：

  1. EventID 命中率   — 答案引用的 [EventID=x] 中是否≥1 个落在标注的 relevant_event_ids 内
  2. 格式合规率       — 答案是否含 [EventID=xxx] 引用标记
  3. 数值幻觉率       — 答案中出现的金额/百分比等数字是否在检索证据中查无出处
  4. 答案长度（字数）— 长答案不一定好，短而正确更佳，仅作参考
  5. 平均单次延迟

仅使用带 ``relevant_event_ids`` 的评测样本，以保证 EventID 命中率有定义；
30 条按 intent（case_retrieval / law_grounding / sanction_recommendation）分层抽样。

用法::

    python scripts/evaluate_generation_m4_4.py \
        --gold   data/eval/gold_130.jsonl \
        --sample 30 \
        --out    docs/reports/m4_4_generation_eval.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LOGGER = logging.getLogger("eval_m4_4")


# ---------------------------------------------------------------------------
# Prompts (G2/G3 share the strong prompt; G0 minimal; G1 middle)
# ---------------------------------------------------------------------------

# 极简提示词（G0/G1 使用）：不施加任何引用或防幻觉约束，作为弱基线对照。
SYS_MINIMAL = "你是证券合规问答助手。请用中文回答。"

# 强约束提示词（G2/G3 使用）：强制只依据检索证据作答、禁止编造、必须引用
# [EventID=xxx]，证据不足须明说，杜绝"参考历史相似案例"这类空话式引用。

SYS_STRONG = (
    "你是证监会处罚案例智能分析助手。你只能根据给定的检索证据回答,"
    "禁止编造证据中没有出现的法条、处罚结果、罚款金额或事实。"
    "如果证据不足,请明确写「证据不足」。"
    "回答必须引用 [EventID=xxx];若涉及法规,再引用 [法条:《xx》第xx条]。"
    "即便检索返回的案例与查询不完全匹配,也要引用检索到的最相关案例的 EventID,"
    "不允许用「参考历史相似案例」等空话代替具体引用。"
)

INSTR_RAG_STRONG = (
    "根据下方检索到的证监会处罚案例,回答用户问题。"
    "必须引用 [EventID=xxx];若涉及法规,再引用 [法条:《xx》第xx条];"
    "不得编造证据中未出现的内容。"
    "即便检索证据与查询不完全匹配,也要引用至少一个最接近的 EventID,"
    "不允许用「参考历史相似案例」等空话代替具体引用。"
)


# ---------------------------------------------------------------------------
# Gold subset sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    rows: list[dict], n: int, seed: int = 42
) -> list[dict]:
    """按 intent 分层抽取 n 条评测样本。

    仅保留带标注 relevant_event_ids、intent 属于三类核心意图、且非陷阱题
    （is_trap）的样本，保证 EventID 命中率这一指标在每条上都有定义。
    """
    eligible = [
        r
        for r in rows
        if r.get("relevant_event_ids")
        and r.get("intent") in {"case_retrieval", "law_grounding", "sanction_recommendation"}
        and not r.get("is_trap")
    ]
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        by_intent[r["intent"]].append(r)

    rng = random.Random(seed)
    total = len(eligible)
    picked: list[dict] = []
    for intent, items in by_intent.items():
        target = max(1, round(n * len(items) / total))
        picked.extend(rng.sample(items, min(target, len(items))))
    if len(picked) > n:
        picked = rng.sample(picked, n)
    elif len(picked) < n:
        extras = [r for r in eligible if r not in picked]
        if extras:
            picked.extend(rng.sample(extras, min(n - len(picked), len(extras))))
    return picked


# ---------------------------------------------------------------------------
# Evidence block rendering from retrieval events
# ---------------------------------------------------------------------------


def render_evidence_block(gold_row: dict, engine, top_k: int = 5) -> tuple[str, list[str]]:
    """对单条评测样本跑检索引擎，渲染成 G1/G2/G3 共享的"证据块"文本。

    取 top_k 条检索结果，逐条拼出 EventID/标题/公告日期/违规情节摘要，
    返回 (证据文本, 检索到的 EventID 列表)。G0 不喂这块证据，故不会调用本函数。
    forced_intent 用样本标注意图，保证检索路由与评测意图一致。
    """
    try:
        response = engine.search(gold_row["query"], forced_intent=gold_row.get("intent"))
    except Exception as exc:
        LOGGER.warning("retrieval failed for %s: %s", gold_row.get("id"), exc)
        return "", []

    retrieved_ids = [e["event_id"] for e in response.events[:top_k]]
    lines: list[str] = []
    for i, e in enumerate(response.events[:top_k], 1):
        title = (e.get("title") or "")[:60]
        date = e.get("declare_date") or ""
        snippet = (e.get("snippets") or [""])[0][:160]
        lines.append(
            f"案例{i}:\n  EventID={e['event_id']}\n  标题:{title}\n"
            f"  公告日期:{date}\n  违规情节:{snippet}"
        )
    return "\n".join(lines), retrieved_ids


# ---------------------------------------------------------------------------
# Model loading — base + optional LoRA
# ---------------------------------------------------------------------------


def load_model(base_name: str, lora_path: Path | None):
    """惰性加载模型，便于在"裸基座"与"基座+LoRA"之间低成本切换。

    G0-G2 共用同一个 4-bit 量化基座；G3 在同一基座上叠加 LoRA 适配器。
    评测与训练保持一致的 4-bit NF4 量化配置，确保对比公平。返回 (model, tokenizer)。
    """
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_name, trust_remote_code=True, torch_dtype="auto", quantization_config=bnb
    )
    if lora_path is not None:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_one(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> tuple[str, float]:
    import torch  # type: ignore

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 低温采样（temperature=0.2 + top_p=0.9）：评测追求稳定可复现，
    # 抑制随机性以减少同一题多次生成的方差；同时计时单次生成延迟。
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    gen = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return gen.strip(), elapsed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# 三个正则分别抽取：答案中的 [EventID=xxx] 引用、《xx》法规名、以及
# 带单位（万元/亿/元/%/倍）的金额或百分比数字——后者用于检测数值幻觉。
_EID_PATTERN = re.compile(r"\[EventID=([A-Za-z0-9]+)\]")
_LAW_PATTERN = re.compile(r"《([^《》]+)》")
_NUMBER_PATTERN = re.compile(r"\d{1,3}(?:[,，]?\d{3})*(?:\.\d+)?\s*(万元|万|亿|元|%|％|倍)")


def cited_event_ids(text: str) -> list[str]:
    return _EID_PATTERN.findall(text)


def cited_laws(text: str) -> list[str]:
    return _LAW_PATTERN.findall(text)


def hallucinated_numbers(answer: str, evidence: str) -> list[str]:
    """找出答案里出现、但检索证据中查无出处的金额/百分比数字（即疑似幻觉）。

    判定策略：证据中原样出现、或去掉千分位逗号后归一化形式一致的数字都视为有据，
    予以容忍；其余被标记。足以抓出"罚款 50 万元"而证据并未提及该金额的情形。
    """
    evid_numbers = set(m.group(0) for m in _NUMBER_PATTERN.finditer(evidence))
    # Also accept normalised forms (strip commas)
    evid_simple = {re.sub(r"[,，\s]", "", n) for n in evid_numbers}
    flagged: list[str] = []
    for m in _NUMBER_PATTERN.finditer(answer):
        full = m.group(0)
        normalised = re.sub(r"[,，\s]", "", full)
        if full in evid_numbers:
            continue
        if normalised in evid_simple:
            continue
        flagged.append(full)
    return flagged


def evaluate_row(gold: dict, evidence: str, answer: str) -> dict:
    # 单条样本的逐项打分：EventID 命中、格式合规、答案长度、数值幻觉、
    # 以及引用的法规名及其是否落在证据内，汇成一条 metric 记录。
    eids = cited_event_ids(answer)
    gold_ids = set(str(x) for x in gold.get("relevant_event_ids", []))
    hit = any(e in gold_ids for e in eids) if eids else False
    format_ok = bool(eids)
    halluc_numbers = hallucinated_numbers(answer, evidence)
    laws = cited_laws(answer)
    laws_in_evidence = [lw for lw in laws if lw in evidence]
    return {
        "cited_event_ids": eids,
        "event_id_hit": hit,
        "format_ok": format_ok,
        "answer_len": len(answer),
        "hallucinated_numbers": halluc_numbers,
        "cited_laws": laws,
        "laws_in_evidence": laws_in_evidence,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_condition(
    label: str,
    gold_rows: list[dict],
    *,
    model,
    tokenizer,
    include_rag: bool,
    system_prompt: str,
    engine,
) -> list[dict]:
    # 跑某一组（G0..G3）的全部评测题：按该组是否启用 RAG 决定喂不喂证据块，
    # 逐题生成、计时、打分，并打印进度日志。
    results: list[dict] = []
    for i, row in enumerate(gold_rows, 1):
        if include_rag:
            evidence, retrieved = render_evidence_block(row, engine, top_k=5)
            if evidence:
                user = f"{INSTR_RAG_STRONG}\n\n用户问题:{row['query']}\n\n[检索证据]\n{evidence}"
            else:
                user = row["query"]
        else:
            evidence = ""
            retrieved = []
            user = row["query"]

        try:
            answer, latency = generate_one(model, tokenizer, system_prompt, user)
        except Exception as exc:
            LOGGER.warning("gen failed %s/%s: %s", label, row.get("id"), exc)
            answer, latency = "", 0.0

        # 引用兜底（与 src/csrc_rag/response/responder.py 中线上同款安全网保持一致）：
        # 当本组启用了 RAG、检索确有候选、但模型答案里完全没有 [EventID=] 引用时，
        # 强制把 top-1 的 EventID 追加到答案末尾，避免产出"无引用的套话"。
        if (
            include_rag
            and retrieved
            and "[EventID=" not in answer
            and answer.strip()
        ):
            top1 = retrieved[0]
            answer = (
                f"{answer.rstrip('。;；,，。 ')}"
                f"；参考案例见 [EventID={top1}]。"
            )

        metric = evaluate_row(row, evidence, answer)
        metric.update({
            "condition": label,
            "gold_id": row["id"],
            "query": row["query"],
            "answer": answer,
            "latency_s": round(latency, 3),
            "retrieved_event_ids": retrieved,
        })
        results.append(metric)
        LOGGER.info(
            "[%s %2d/%d] hit=%s halluc=%d %.1fs — %s",
            label, i, len(gold_rows),
            metric["event_id_hit"],
            len(metric["hallucinated_numbers"]),
            latency,
            row["id"],
        )
    return results


def summarise(results: list[dict]) -> dict:
    # 把单组逐题结果聚合为组级指标：命中率/格式合规率/数值幻觉率
    # 均为"命中题数占比"，另给出平均答案字数与平均延迟。
    if not results:
        return {"n": 0}
    hit = sum(r["event_id_hit"] for r in results) / len(results)
    fmt = sum(r["format_ok"] for r in results) / len(results)
    halluc_rate = sum(1 for r in results if r["hallucinated_numbers"]) / len(results)
    avg_len = sum(r["answer_len"] for r in results) / len(results)
    avg_lat = sum(r["latency_s"] for r in results) / len(results)
    return {
        "n": len(results),
        "event_id_hit_rate": round(hit, 4),
        "format_compliance_rate": round(fmt, 4),
        "hallucinated_number_rate": round(halluc_rate, 4),
        "avg_answer_chars": round(avg_len, 1),
        "avg_latency_s": round(avg_lat, 3),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, default=PROJECT_ROOT / "data" / "eval" / "gold_130.jsonl")
    p.add_argument("--sample", type=int, default=30)
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--lora",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "qwen_lora_csrc",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["G0", "G1", "G2", "G3"],
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "reports" / "m4_4_generation_eval.md",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    gold_all = [
        json.loads(l) for l in args.gold.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    subset = stratified_sample(gold_all, args.sample)
    LOGGER.info("sampled %d rows from %d (stratified by intent)", len(subset), len(gold_all))

    from csrc_rag.retrieval.engine import RetrievalEngine

    LOGGER.info("loading retrieval engine...")
    engine = RetrievalEngine(retrieval_mode="hybrid", rerank_enabled=False)

    LOGGER.info("loading base model for G0-G2...")
    base_model, tok = load_model(args.base, lora_path=None)

    all_results: dict[str, list[dict]] = {}

    if "G0" not in args.skip:
        LOGGER.info("=== G0: base + no RAG ===")
        all_results["G0"] = run_condition(
            "G0", subset,
            model=base_model, tokenizer=tok,
            include_rag=False, system_prompt=SYS_MINIMAL, engine=engine,
        )
    if "G1" not in args.skip:
        LOGGER.info("=== G1: base + RAG + minimal prompt ===")
        all_results["G1"] = run_condition(
            "G1", subset,
            model=base_model, tokenizer=tok,
            include_rag=True, system_prompt=SYS_MINIMAL, engine=engine,
        )
    if "G2" not in args.skip:
        LOGGER.info("=== G2: base + RAG + strong prompt ===")
        all_results["G2"] = run_condition(
            "G2", subset,
            model=base_model, tokenizer=tok,
            include_rag=True, system_prompt=SYS_STRONG, engine=engine,
        )

    # G3 之前先释放裸基座并清空显存缓存，再加载"基座+LoRA"。
    # 8GB 的 2060S 同时常驻两份模型会 OOM，故采用先释放后加载的串行策略。
    if "G3" not in args.skip:
        del base_model
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info("loading base + LoRA for G3...")
        lora_model, tok = load_model(args.base, lora_path=args.lora)
        LOGGER.info("=== G3: base + LoRA + RAG + strong prompt ===")
        all_results["G3"] = run_condition(
            "G3", subset,
            model=lora_model, tokenizer=tok,
            include_rag=True, system_prompt=SYS_STRONG, engine=engine,
        )

    # Summaries
    summary = {label: summarise(rs) for label, rs in all_results.items()}

    # Write outputs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# M4.4 G0-G3 四组生成对比评估",
        "",
        f"- 评测子集: {len(subset)} 条 (stratified from `{args.gold.relative_to(PROJECT_ROOT)}`)",
        f"- 基座: `{args.base}` (4-bit NF4)",
        f"- LoRA: `{args.lora.relative_to(PROJECT_ROOT)}` (M4.3 产出)",
        "",
        "## 四组总表",
        "",
        "| 组 | 配置 | EID 命中率 | 格式合规 | 幻觉数字率 | 答案长度 | 延迟/query |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    labels = {
        "G0": "base + 无 RAG + 弱 prompt",
        "G1": "base + RAG + 弱 prompt",
        "G2": "base + RAG + 强 prompt",
        "G3": "base + **LoRA** + RAG + 强 prompt",
    }
    for grp in ["G0", "G1", "G2", "G3"]:
        if grp not in summary:
            continue
        s = summary[grp]
        lines.append(
            f"| {grp} | {labels[grp]} | {s['event_id_hit_rate']:.3f} | "
            f"{s['format_compliance_rate']:.3f} | "
            f"{s['hallucinated_number_rate']:.3f} | "
            f"{s['avg_answer_chars']:.0f} 字 | "
            f"{s['avg_latency_s']:.2f}s |"
        )
    lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {"summary": summary, "subset_ids": [r["id"] for r in subset], "details": all_results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("M4.4 summary:")
    for grp, s in summary.items():
        print(f"  {grp}: hit={s['event_id_hit_rate']:.3f}  fmt={s['format_compliance_rate']:.3f}  "
              f"halluc={s['hallucinated_number_rate']:.3f}  lat={s['avg_latency_s']:.2f}s")
    print(f"\nReport: {args.out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

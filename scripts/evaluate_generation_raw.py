"""M4.4: G0-G3 four-group generation ablation.

Four conditions on the same 30-question eval subset:

  G0  Qwen-0.5B base          · no RAG · minimal system prompt
  G1  Qwen-0.5B base          · + RAG evidence · minimal system prompt
  G2  Qwen-0.5B base          · + RAG evidence · strong-grounding prompt
  G3  Qwen-0.5B + LoRA(M4.3)  · + RAG evidence · strong-grounding prompt

Metrics per group:

  1. EventID hit rate     — whether ≥1 cited [EventID=x] ∈ gold.relevant_event_ids
  2. Format compliance    — whether the answer contains [EventID=xxx]
  3. Hallucinated numbers — whether the answer mentions a number (金额/百分比)
                             that is NOT present in the evidence
  4. Answer length        — chars (长答案不一定好,短而正确更佳)
  5. Avg latency (ms)

Only eval rows with ``relevant_event_ids`` are used so EventID hit rate is defined.
30 rows stratified by intent (case_retrieval / law_grounding / sanction_recommendation).

Usage::

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

SYS_MINIMAL = "你是证券合规问答助手。请用中文回答。"

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
    """Pick n rows stratified by ``intent``, only keeping rows with gold IDs."""
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
    """Run the retrieval engine for gold_row and render an evidence block
    that the G1/G2/G3 conditions all see.

    Returns (evidence_text, retrieved_event_ids). For G0 this block is
    NOT supplied (returns empty retrieval).
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
    """Lazy load so we can swap (base) vs (base+lora) cheaply.

    Returns (model, tokenizer).
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

_EID_PATTERN = re.compile(r"\[EventID=([A-Za-z0-9]+)\]")
_LAW_PATTERN = re.compile(r"《([^《》]+)》")
_NUMBER_PATTERN = re.compile(r"\d{1,3}(?:[,，]?\d{3})*(?:\.\d+)?\s*(万元|万|亿|元|%|％|倍)")


def cited_event_ids(text: str) -> list[str]:
    return _EID_PATTERN.findall(text)


def cited_laws(text: str) -> list[str]:
    return _LAW_PATTERN.findall(text)


def hallucinated_numbers(answer: str, evidence: str) -> list[str]:
    """Numbers (amounts / percentages) in the answer that aren't in evidence.

    Heuristic: we tolerate numbers that appear verbatim or whose integer
    value appears in evidence. Strict enough to flag "罚款 50 万元" when
    evidence doesn't mention that amount.
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

        # B2 fix DISABLED for raw evaluation — we want to see native model output
        # without post-processing, to measure what the model actually learned.
        # if (
        #     include_rag
        #     and retrieved
        #     and "[EventID=" not in answer
        #     and answer.strip()
        # ):
        #     top1 = retrieved[0]
        #     answer = (
        #         f"{answer.rstrip('。;；,，。 ')}"
        #         f"；参考案例见 [EventID={top1}]。"
        #     )

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

    # Free base model, load LoRA for G3
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

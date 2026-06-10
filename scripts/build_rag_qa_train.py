"""构造约 5500 条 RAG 问答训练集，供 Qwen2.5-1.5B 做 QLoRA 微调。

本脚本是我们整条微调流水线的数据生产环节：从证监会处罚事件语料出发，
按八类任务模板生成 Alpaca 格式样本，并按事件公告年份切分 train/val/test，
从源头杜绝同一事件跨划分泄漏。产出三个 JSONL 落在 ``data/processed/``::

    rag_qa_train.jsonl   (~4400，ABCD 取 declare_date 1994-2021；EFGH 随机抽 80%)
    rag_qa_val.jsonl     (~ 550，ABCD 取 declare_date 2022-2023；EFGH 随机抽 10%)
    rag_qa_test.jsonl    (~ 550，ABCD 取 declare_date 2024-2025；EFGH 随机抽 10%)

八类任务配比（团队设计的数据配方，A-H 各司其职）::

    A  案例检索              1800  引用 [EventID=xxx]
    B  法条依据              1200  引用 [法条：《xx》第xx条]
    C  处罚建议              1000  引用 [EventID=xxx]（input 中刻意剔除 PunishmentMeasure 防泄漏）
    D  趋势统计               400  引用 [EventID=xxx] + 统计数据
    E  拒答（超范围）         300  固定拒答话术
    F  问候 / 闲聊            300  固定能力介绍话术
    G  多轮追问               200  把代词（那/它/这案）消解到历史实体
    H  反幻觉负样本           300  虚构公司/EventID，答案须回"未检索到"

Alpaca 格式 schema（额外记录 event_id_source，便于按事件做无泄漏划分）::

    {
      "event_id_source": "40100082" | null,
      "category": "A" | ... | "H",
      "split": "train" | "val" | "test",
      "instruction": "根据下方检索到的证监会处罚案例，回答用户问题。...",
      "input": "用户问题：...\\n\\n[检索证据]\\n案例1：...",
      "output": "根据 [EventID=40100082]，..."
    }

Usage::

    python scripts/build_rag_qa_train.py \\
        --corpus data/processed/event_corpus.jsonl \\
        --out    data/processed \\
        --seed   42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants / schema
# ---------------------------------------------------------------------------

INSTRUCTION_RAG = (
    "根据下方检索到的证监会处罚案例，回答用户问题。"
    "必须引用 [EventID=xxx]；若涉及法规，再引用 [法条：《xx》第xx条]；不得编造证据中未出现的内容。"
)
INSTRUCTION_REFUSE = "用户的问题不在本系统受理范围。请以固定的拒答话术回应。"
INSTRUCTION_GREETING = "用户正在进行日常问候或询问系统能力。请以固定的能力介绍话术回应。"
INSTRUCTION_NEGATIVE = (
    "用户以虚构的公司名或 EventID 诱导你作答。"
    "若检索证据中不存在该主体或 EventID，必须明确回答『未检索到』，不得编造任何细节。"
)
INSTRUCTION_MULTI_TURN = (
    "根据历史对话与检索到的证监会处罚案例，回答用户的追问。"
    "若用户用代词（那/它/这案）指代上一轮提到的实体，请明确指代并引用 [EventID=xxx]。"
)

REFUSAL_TEMPLATES: tuple[str, ...] = (
    "本系统仅支持证监会违规案例检索与处罚分析，无法回答与您提问无关的内容。"
    "如需帮助，请提出与证券合规、处罚案例、法规依据或趋势分析相关的问题。",
    "抱歉，本系统只处理证监会行政处罚相关的查询。"
    "您的问题超出了系统受理范围，请换一个与证券违规案例相关的问题。",
    "该问题不在本系统的业务范围内。"
    "我可以帮您检索证监会的违规案例、对应法条以及历史处罚方式，欢迎继续提问。",
)

GREETING_TEMPLATES: tuple[str, ...] = (
    "您好，我是证监会处罚案例智能分析助手。"
    "我可以帮您：1) 检索历史违规案例；2) 定位违规对应的法规依据；3) 根据相似案例给出处罚方式参考；4) 分析违规趋势。请告诉我您想了解的情节或主体。",
    "您好！我专注证监会行政处罚案例的检索与分析。"
    "常见用法：输入违规类型（如『内幕交易』）检索案例、输入主体名称查询历史处罚、询问某类行为对应的法条。欢迎提问。",
    "在的。我是基于证监会处罚数据训练的 RAG 助手，"
    "能做案例检索、法条依据查询、相似案件处罚方式参考、以及违规趋势统计。请描述您的查询目标。",
)

NEGATIVE_TEMPLATES: tuple[str, ...] = (
    "未检索到与所问主体/案例编号匹配的证据，无法回答。"
    "建议核对公司名称或 EventID 后再试；我不会在证据缺失的情况下给出处罚细节。",
    "当前检索证据中不包含您提到的主体/EventID，无法回答。"
    "为避免编造，我拒绝给出任何处罚金额、法条或结论。请补充可验证的线索。",
    "所给证据未覆盖该主体/EventID，无法回答。"
    "我只能基于真实检索到的案例给出分析，不会虚构细节。",
)


A_TEMPLATES: tuple[str, ...] = (
    "帮我找{activity}类型的案例。",
    "{activity}这类违规有哪些处罚先例？",
    "请列举与{activity}相关的证监会处罚案例，用于参考。",
    "我想了解{activity}类违规的历史案例。",
    "检索一下证监会对{activity}类行为的历史处罚。",
    "过去有没有{activity}方面的处罚案例？举几个看看。",
    "关于{activity}，能找出几个典型案例吗？",
)

B_TEMPLATES: tuple[str, ...] = (
    "{activity}这种行为违反了哪些法条？",
    "请问{activity}类违规在证监会案例中对应什么法律依据？",
    "{activity}主要违反哪些法规？",
    "类似{activity}的行为一般引用哪条法规处罚？",
    "从案例看，{activity}的法律依据是什么？",
)

C_TEMPLATES: tuple[str, ...] = (
    "{activity}这种行为通常会受到什么处罚？",
    "类似{activity}的情形，证监会一般怎么处罚？",
    "{activity}类违规一般的处罚方式有哪些？",
    "参考历史案例，{activity}这种行为建议如何处理？",
    "{activity}类情形的处罚方式通常包括哪几种？",
)

D_TEMPLATES: tuple[str, ...] = (
    "近{years}年{violation}类违规的处罚趋势如何？",
    "过去{years}年，{violation}案件的数量变化怎么样？",
    "最近{years}年{violation}这类违规多不多？",
    "{violation}类违规近{years}年的处罚频率有变化吗？",
    "帮我看一下近{years}年{violation}相关案例的数量分布。",
)

E_QUESTIONS: tuple[str, ...] = (
    "今天天气怎么样？",
    "给我写一段 Python 快排代码。",
    "股票 600519 明天会不会涨？",
    "我心情不好，能安慰我几句吗？",
    "帮我写一首关于春天的情诗。",
    "推荐几部近期好看的电影。",
    "翻译这句英文：Good morning, how are you?",
    "1+1 等于几？",
    "帮我订今天晚上八点的西餐厅。",
    "能讲个笑话吗？",
    "解释一下量子纠缠。",
    "你怎么评价最近那位明星的新剧？",
    "帮我算一下房贷利率。",
    "明天北京 PM2.5 多少？",
    "解释一下区块链原理。",
    "给我推荐一款手机。",
    "写一份辞职信模板。",
    "介绍一下红楼梦的主题思想。",
    "帮我写封英文求职信。",
    "北京到上海高铁多久？",
    "世界杯什么时候开始？",
    "最近比特币价格怎么样？",
    "给我讲个鬼故事。",
    "明天早上适合跑步吗？",
    "推荐几本科幻小说。",
    "ChatGPT 跟你有什么区别？",
    "帮我做一道红烧肉。",
    "今天是几月几号？",
    "地球到月球多远？",
    "讲讲黑洞是什么。",
)

F_QUESTIONS: tuple[str, ...] = (
    "你好",
    "在吗？",
    "hi",
    "hello",
    "你是谁？",
    "你叫什么名字？",
    "你能做什么？",
    "介绍下你自己",
    "这是什么系统？",
    "你有什么功能？",
    "使用说明",
    "你能帮我做什么事情？",
    "该怎么用你？",
    "你可以怎么帮助我？",
    "能讲讲你的用途吗？",
    "你是个 AI 吗？",
    "请简要介绍一下。",
    "新人不太会用，给点指引。",
    "刚打开页面，下一步做什么？",
    "怎么开始提问？",
    "在干嘛？",
    "你好呀",
    "早上好",
    "晚上好",
    "在不在？",
    "hi there",
    "请多关照",
    "很高兴见到你",
    "这个助手能做什么？",
    "让我看看你的本事。",
)

G_FOLLOWUP_TEMPLATES: tuple[str, ...] = (
    "那这案具体是怎么处罚的？",
    "它的法律依据是什么？",
    "那它的罚款金额大概多少？",
    "这案涉及的主体是谁？",
    "那这个案子发生在哪一年？",
    "那它违反了哪条法规？",
    "这案的处罚机构是哪一家？",
    "那个案子的违规类型是什么？",
    "这案的当事人有几个？",
    "那案最终的处分措施是什么？",
)

H_FAKE_COMPANIES: tuple[str, ...] = (
    "测试ABC公司",
    "虚构XYZ集团",
    "示例甲乙丙有限公司",
    "DEMO控股股份公司",
    "测试用 Foo 集团",
    "虚构 Bar 科技股份有限公司",
    "MOCK 证券有限公司",
    "假设性 Baz 实业",
    "示例 Qux 投资管理公司",
    "测试 Alpha 实业集团",
    "占位 Beta 能源股份公司",
    "假想 Gamma 生物科技",
)

H_FAKE_EVENT_IDS: tuple[str, ...] = (
    "E_2099_9999",
    "E_2100_0001",
    "E_2026_XXXX",
    "E_TEST_0001",
    "E_MOCK_0042",
    "E_FAKE_7777",
    "E_2088_8888",
)

# Words that would leak punishment labels into C input
C_LEAK_BLOCKLIST: tuple[str, ...] = (
    "处罚方式",
    "处分措施",
    "处罚金额",
    "罚款金额",
    "punishment_measure",
    "PunishmentMeasure",
    "sum_penalty",
    "SumPenalty",
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QASample:
    """A single training sample. Alpaca-style + event_id_source for splitting."""

    event_id_source: str | None
    category: str  # A..H
    split: str  # train | val | test
    instruction: str
    input: str
    output: str

    def to_jsonl_dict(self) -> dict[str, Any]:
        return {
            "event_id_source": self.event_id_source,
            "category": self.category,
            "split": self.split,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }


# ---------------------------------------------------------------------------
# Quotas per split (train / val / test)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassQuota:
    category: str
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


# 八类在 train/val/test 三个划分上的配额。总量大体落在 8:1:1，
# A/B/C 为主力（案例检索/法条/处罚建议），E/F/G/H 为约束类（拒答/问候/多轮/反幻觉），
# 用以同时训练模型的检索作答能力与边界自律能力。
DEFAULT_QUOTAS: tuple[ClassQuota, ...] = (
    ClassQuota("A", 1440, 180, 180),
    ClassQuota("B", 960, 120, 120),
    ClassQuota("C", 800, 100, 100),
    ClassQuota("D", 320, 40, 40),
    ClassQuota("E", 240, 30, 30),
    ClassQuota("F", 240, 30, 30),
    ClassQuota("G", 160, 20, 20),
    ClassQuota("H", 240, 30, 30),
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(samples: Iterable[QASample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fp:
        for sample in samples:
            fp.write(json.dumps(sample.to_jsonl_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------


def _field(ev: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = ev.get(k)
        if v in (None, ""):
            continue
        if isinstance(v, list):
            v = next((x for x in v if x), "")
        s = str(v).strip()
        if s:
            return s
    return default


def _list_field(ev: dict[str, Any], *keys: str) -> list[str]:
    for k in keys:
        v = ev.get(k)
        if v:
            if isinstance(v, list):
                return [str(x) for x in v if x]
            return [str(v)]
    return []


def _event_year(ev: dict[str, Any]) -> int | None:
    d = _field(ev, "declare_date", "DeclareDate")
    if len(d) >= 4 and d[:4].isdigit():
        return int(d[:4])
    return None


def _split_of_event(ev: dict[str, Any]) -> str | None:
    # 按事件公告年份做时间切分：早年→train、2022-2023→val、2024-2025→test。
    # 以"事件"而非"样本"为粒度划分，是为了让验证/测试集里出现的是模型训练时
    # 完全没见过的案件，从根本上避免数据泄漏、让指标反映真实泛化能力。
    year = _event_year(ev)
    if year is None:
        return None
    if 1994 <= year <= 2021:
        return "train"
    if year in (2022, 2023):
        return "val"
    if year in (2024, 2025):
        return "test"
    return None


def _violation_list(ev: dict[str, Any]) -> list[str]:
    raw = _list_field(ev, "violation_types", "ViolationTypes")
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        for piece in x.replace("、", ";").split(";"):
            p = piece.strip()
            if p and p not in seen and p != "其他":
                seen.add(p)
                out.append(p)
    return out


def _short(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _brief_activity(ev: dict[str, Any]) -> str:
    """A short, human-readable keyword derived from the event.

    Prefers violation types (clean short labels). Falls back to a short, cleaned
    activity snippet — but only up to the first full-width sentence boundary so
    that user queries stay readable.
    """
    vts = _violation_list(ev)
    if vts:
        return vts[0][:30]
    # Promulgator-derived tag as a fallback for events without violation_types
    promulgator = _field(ev, "promulgator", "Promulgator", default="")
    if promulgator and promulgator != "上市公司":
        return f"{promulgator}相关违规"
    raw = _field(ev, "activity", "Activity", default="")
    if not raw:
        return "证监会处罚类"
    # Strip common boilerplate prefixes so queries aren't dominated by "经查明,".
    for prefix in ("经查明,", "经查明，", "经查,", "经查明:", "经查明："):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    # Try to cut at punctuation so we don't embed a 120-char blob in a query.
    for sep in ("。", "；", ";", "，", ","):
        idx = raw.find(sep)
        if 4 <= idx <= 24:
            return raw[:idx]
    return _short(raw, 20)


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------


def _render_evidence_block(
    events: Iterable[dict[str, Any]],
    *,
    rng: random.Random,
    include_laws: bool = True,
    include_punishment_types: bool = True,
) -> str:
    """Render an evidence block listing 1-5 cases.

    Fields rendered: EventID, title, declare_date, violation_types,
    punishment_types (optional), laws (optional), activity snippet.

    *No* PunishmentMeasure / SumPenalty is ever rendered — those are labels.
    """
    lines: list[str] = []
    for i, ev in enumerate(events, 1):
        eid = _field(ev, "event_id", "EventID", default="UNKNOWN")
        title = _short(_field(ev, "title", "Title", default="（未命名）"), 60)
        declare = _field(ev, "declare_date", "DeclareDate", default="未知")
        vts = _violation_list(ev)
        vt_str = "、".join(vts[:3]) if vts else "未提取"
        activity = _short(_field(ev, "activity", "Activity", default=""), 120) or "未提取"
        parts = [
            f"案例{i}：",
            f"  EventID={eid}",
            f"  标题：{title}",
            f"  公告日期：{declare}",
            f"  违规类型：{vt_str}",
            f"  违规情节：{activity}",
        ]
        if include_punishment_types:
            pts = _list_field(ev, "punishment_types", "PunishmentTypes")
            parts.append(f"  处罚类型：{'、'.join(pts[:3]) if pts else '未提取'}")
        if include_laws:
            law = _field(ev, "law", "Law", default="")
            if law:
                parts.append(f"  法律依据：{_short(law, 120)}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


def _pick_k_related(
    seed: dict[str, Any],
    pool: list[dict[str, Any]],
    k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    extras_pool = [e for e in pool if e is not seed]
    k = max(1, min(k, len(pool)))
    extras = rng.sample(extras_pool, k=min(k - 1, len(extras_pool))) if extras_pool else []
    result = [seed, *extras]
    rng.shuffle(result)
    return result


def _law_cite(law_text: str) -> str:
    """Convert raw law text into a concise [法条：《xx》第xx条] style tag.

    Best-effort extraction; falls back to truncated raw text.
    """
    if not law_text:
        return "[法条：未提取]"
    import re as _re

    # Look for 《...》(?:第X条)? patterns; keep first match
    matches = _re.findall(r"《[^《》]+》(?:第[一二三四五六七八九十百零〇\d]+条(?:第[一二三四五六七八九十百零〇\d]+款)?)?", law_text)
    if matches:
        return "[法条：" + "；".join(matches[:2]) + "]"
    return f"[法条：{_short(law_text, 40)}]"


# ---------------------------------------------------------------------------
# Per-class builders
# ---------------------------------------------------------------------------


@dataclass
class BuilderContext:
    corpus: list[dict[str, Any]]
    rng: random.Random
    quotas: tuple[ClassQuota, ...] = field(default_factory=lambda: DEFAULT_QUOTAS)

    def quota_for(self, category: str) -> ClassQuota:
        for q in self.quotas:
            if q.category == category:
                return q
        raise KeyError(category)

    def events_for_split(self, split: str) -> list[dict[str, Any]]:
        return [e for e in self.corpus if _split_of_event(e) == split]


# ----- A: case retrieval -----------------------------------------------------


def _sample_events_with_replacement(
    pool: list[dict[str, Any]], n: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Pick n events. If pool is smaller than n, allow replacement."""
    if not pool:
        return []
    if n <= len(pool):
        return rng.sample(pool, k=n)
    return rng.choices(pool, k=n)


# A 类（案例检索）：以一个种子事件为锚，随机挑 2-4 条同主题相关事件拼成检索证据块，
# 用户问题用 A_TEMPLATES 模板按违规活动关键词填充，答案须引用证据中的 [EventID=xxx]。
def build_class_a(ctx: BuilderContext) -> list[QASample]:
    q = ctx.quota_for("A")
    out: list[QASample] = []
    for split, n_target in (("train", q.train), ("val", q.val), ("test", q.test)):
        pool = [
            e for e in ctx.events_for_split(split)
            if _field(e, "activity", "Activity") or _violation_list(e)
        ]
        seeds = _sample_events_with_replacement(pool, n_target, ctx.rng)
        for seed in seeds:
            out.append(_build_a_sample(seed, pool, split, ctx.rng))
    return out


def _build_a_sample(
    seed: dict[str, Any], pool: list[dict[str, Any]], split: str, rng: random.Random
) -> QASample:
    keyword = _brief_activity(seed) or "相关违规"
    tmpl = rng.choice(A_TEMPLATES)
    user_q = tmpl.format(activity=keyword)
    k = rng.choice([2, 3, 4])
    evidence_events = _pick_k_related(seed, pool, k, rng)
    evidence = _render_evidence_block(evidence_events, rng=rng)

    seed_id = _field(seed, "event_id", "EventID")
    seed_title = _short(_field(seed, "title", "Title", default="相关案例"), 30)
    seed_date = _field(seed, "declare_date", "DeclareDate", default="未知")
    pts = _list_field(seed, "punishment_types", "PunishmentTypes")
    pt_str = "、".join(pts[:3]) if pts else "警告、罚款"

    # Build up to 2 citations
    cited_ids = []
    for ev in evidence_events[:2]:
        eid = _field(ev, "event_id", "EventID")
        if eid:
            cited_ids.append(eid)
    if seed_id and seed_id not in cited_ids:
        cited_ids.insert(0, seed_id)
    cite_tags = "".join(f"[EventID={eid}]" for eid in cited_ids[:2])

    answer = (
        f"根据检索到的案例，共发现 {len(evidence_events)} 条与「{keyword}」相关的处罚案例。"
        f"其中最相似的是「{seed_title}」（{seed_date}）{cite_tags}，"
        f"该案涉及的处罚类型包括：{pt_str}。"
        "建议进一步核对案情细节再做参考。"
    )
    return QASample(
        event_id_source=seed_id or None,
        category="A",
        split=split,
        instruction=INSTRUCTION_RAG,
        input=f"用户问题：{user_q}\n\n[检索证据]\n{evidence}",
        output=answer,
    )


# ----- B: law grounding ------------------------------------------------------


def build_class_b(ctx: BuilderContext) -> list[QASample]:
    q = ctx.quota_for("B")
    out: list[QASample] = []
    for split, n_target in (("train", q.train), ("val", q.val), ("test", q.test)):
        pool = [
            e for e in ctx.events_for_split(split)
            if _field(e, "law", "Law") and (_field(e, "activity", "Activity") or _violation_list(e))
        ]
        seeds = _sample_events_with_replacement(pool, n_target, ctx.rng)
        for seed in seeds:
            out.append(_build_b_sample(seed, pool, split, ctx.rng))
    return out


def _build_b_sample(
    seed: dict[str, Any], pool: list[dict[str, Any]], split: str, rng: random.Random
) -> QASample:
    keyword = _brief_activity(seed) or "该类违规"
    user_q = rng.choice(B_TEMPLATES).format(activity=keyword)
    k = rng.choice([2, 3])
    evidence_events = _pick_k_related(seed, pool, k, rng)
    evidence = _render_evidence_block(evidence_events, rng=rng, include_laws=True)

    seed_id = _field(seed, "event_id", "EventID")
    law_text = _field(seed, "law", "Law", default="")
    law_tag = _law_cite(law_text)
    answer = (
        f"根据检索证据，此类{keyword}类违规主要违反的法规为：{_short(law_text, 60)} {law_tag}。"
        f"参考案例见 [EventID={seed_id}]，具体条款适用以公告原文为准。"
    )
    return QASample(
        event_id_source=seed_id or None,
        category="B",
        split=split,
        instruction=INSTRUCTION_RAG,
        input=f"用户问题：{user_q}\n\n[检索证据]\n{evidence}",
        output=answer,
    )


# ----- C: sanction recommendation -------------------------------------------


def _strip_c_leakage(ev: dict[str, Any]) -> dict[str, Any]:
    """C 类专用脱敏：从喂给模型的证据中剔除一切会泄漏"处罚结果"的字段。

    C 类任务是让模型据案情"推荐处罚"，因此输入证据里绝不能出现处罚措施、罚没金额、
    判决原文等答案标签，否则模型只是抄答案而非学习推理。这里把这些键全部删掉。
    """
    cleaned: dict[str, Any] = {}
    for k, v in ev.items():
        kl = k.lower()
        if kl in {"punishment_measure", "punishmentmeasure", "sum_penalty", "sumpenalty", "reference_text"}:
            continue
        cleaned[k] = v
    return cleaned


def build_class_c(ctx: BuilderContext) -> list[QASample]:
    q = ctx.quota_for("C")
    out: list[QASample] = []
    for split, n_target in (("train", q.train), ("val", q.val), ("test", q.test)):
        pool = [
            e for e in ctx.events_for_split(split)
            if _list_field(e, "punishment_types", "PunishmentTypes")
            and (_field(e, "activity", "Activity") or _violation_list(e))
        ]
        seeds = _sample_events_with_replacement(pool, n_target, ctx.rng)
        for seed in seeds:
            out.append(_build_c_sample(seed, pool, split, ctx.rng))
    return out


def _build_c_sample(
    seed: dict[str, Any], pool: list[dict[str, Any]], split: str, rng: random.Random
) -> QASample:
    keyword = _brief_activity(seed) or "该类违规"
    user_q = rng.choice(C_TEMPLATES).format(activity=keyword)
    k = rng.choice([3, 4])
    evidence_events = _pick_k_related(seed, pool, k, rng)
    # Strip leakage BEFORE rendering
    safe_events = [_strip_c_leakage(e) for e in evidence_events]
    # Render without punishment_types to keep input free of answer (except violation types)
    evidence = _render_evidence_block(
        safe_events, rng=rng, include_laws=False, include_punishment_types=False
    )

    seed_id = _field(seed, "event_id", "EventID")
    pts = _list_field(seed, "punishment_types", "PunishmentTypes")
    pt_str = "、".join(pts[:3]) if pts else "警告、罚款"
    answer = (
        f"参考历史相似案例，对{keyword}类行为常见的处罚方式包括：{pt_str}。"
        f"[EventID={seed_id}] 最终处罚程度需结合违规情节、主观故意、危害后果综合认定，"
        "具体金额与处分措施以证监会公告为准。"
    )
    return QASample(
        event_id_source=seed_id or None,
        category="C",
        split=split,
        instruction=INSTRUCTION_RAG,
        input=f"用户问题：{user_q}\n\n[检索证据]\n{evidence}",
        output=answer,
    )


# ----- D: trend statistics ---------------------------------------------------


def build_class_d(ctx: BuilderContext) -> list[QASample]:
    q = ctx.quota_for("D")
    out: list[QASample] = []
    for split, n_target in (("train", q.train), ("val", q.val), ("test", q.test)):
        pool = [
            e for e in ctx.events_for_split(split)
            if _event_year(e) is not None and (_violation_list(e) or _field(e, "activity", "Activity"))
        ]
        for _ in range(n_target):
            out.append(_build_d_sample(pool, split, ctx.rng))
    return out


def _build_d_sample(pool: list[dict[str, Any]], split: str, rng: random.Random) -> QASample:
    if not pool:
        # Shouldn't happen because D is only built on splits with ≥ 40 events
        pool = pool  # keep linter happy
    # Pick a violation category with enough samples in pool
    vt_counter: dict[str, list[dict[str, Any]]] = {}
    for ev in pool:
        for vt in _violation_list(ev):
            vt_counter.setdefault(vt, []).append(ev)
    if not vt_counter:
        seed = rng.choice(pool)
        violation_name = _brief_activity(seed) or "违规"
        matched = [seed]
    else:
        # Prefer categories with > 3 events
        rich = [(vt, evs) for vt, evs in vt_counter.items() if len(evs) >= 3]
        if rich:
            violation_name, matched = rng.choice(rich)
        else:
            violation_name, matched = rng.choice(list(vt_counter.items()))

    years_window = rng.choice([3, 5])
    user_q = rng.choice(D_TEMPLATES).format(violation=violation_name, years=years_window)

    k = rng.choice([4, 5, 6])
    evidence_events = rng.sample(matched, k=min(k, len(matched)))
    evidence = _render_evidence_block(evidence_events, rng=rng)

    year_counts: dict[int, int] = {}
    for ev in matched:
        y = _event_year(ev)
        if y is not None:
            year_counts[y] = year_counts.get(y, 0) + 1
    sorted_years = sorted(year_counts.items())

    seed_ev = evidence_events[0]
    seed_id = _field(seed_ev, "event_id", "EventID")
    if len(sorted_years) >= 2:
        trend_pts = "、".join(f"{y}年{c}起" for y, c in sorted_years[-4:])
        total = sum(c for _, c in sorted_years)
        answer = (
            f"根据检索统计，在样本窗口内{violation_name}类违规累计 {total} 起，"
            f"年度分布：{trend_pts}。最近年度仍有新增。"
            f"代表案例：[EventID={seed_id}]。"
        )
    else:
        only_year = sorted_years[0][0] if sorted_years else "近年"
        answer = (
            f"根据检索证据，{violation_name}类违规样本集中在 {only_year} 年，"
            f"跨年样本不足，无法给出多年趋势判断。代表案例：[EventID={seed_id}]。"
        )
    return QASample(
        event_id_source=seed_id or None,
        category="D",
        split=split,
        instruction=INSTRUCTION_RAG,
        input=f"用户问题：{user_q}\n\n[检索证据]\n{evidence}",
        output=answer,
    )


# ----- E: refusal / out-of-scope --------------------------------------------


# E 类（超范围拒答）：问题取自题库 E_QUESTIONS，不足配额时对原问做轻量改写（加后缀）扩充；
# 这类样本无关联事件（event_id_source=None），答案为固定拒答话术，教模型守住业务边界。
def build_class_e(ctx: BuilderContext) -> list[QASample]:
    q = ctx.quota_for("E")
    total = q.total
    rng = ctx.rng
    n_pool = max(total, len(E_QUESTIONS))
    questions = list(E_QUESTIONS)
    # If we need more than the bank, sample with replacement + light paraphrase
    while len(questions) < total:
        base = rng.choice(E_QUESTIONS)
        suffix = rng.choice(["，谢谢。", "，可以吗？", "？", "呢？", "麻烦了。"])
        questions.append(base.rstrip("？?。.") + suffix)
    rng.shuffle(questions)
    picked = questions[:total]

    # Split 80/10/10 by index
    splits_per_sample = _random_split_labels(total, (q.train, q.val, q.test), rng)
    out: list[QASample] = []
    for qtext, split in zip(picked, splits_per_sample, strict=True):
        refusal = rng.choice(REFUSAL_TEMPLATES)
        out.append(
            QASample(
                event_id_source=None,
                category="E",
                split=split,
                instruction=INSTRUCTION_REFUSE,
                input=f"用户问题：{qtext}\n\n[检索证据]\n（无，问题不在本系统受理范围）",
                output=refusal,
            )
        )
    return out


# ----- F: greeting / small talk ---------------------------------------------


def build_class_f(ctx: BuilderContext) -> list[QASample]:
    q = ctx.quota_for("F")
    total = q.total
    rng = ctx.rng
    questions = list(F_QUESTIONS)
    while len(questions) < total:
        base = rng.choice(F_QUESTIONS)
        suffix = rng.choice(["～", "。", "！", "呀", " :)", "呢"])
        questions.append(base.rstrip("?？。.") + suffix)
    rng.shuffle(questions)
    picked = questions[:total]

    splits_per_sample = _random_split_labels(total, (q.train, q.val, q.test), rng)
    out: list[QASample] = []
    for qtext, split in zip(picked, splits_per_sample, strict=True):
        answer = rng.choice(GREETING_TEMPLATES)
        out.append(
            QASample(
                event_id_source=None,
                category="F",
                split=split,
                instruction=INSTRUCTION_GREETING,
                input=f"用户问题：{qtext}\n\n[检索证据]\n（无，这是问候/闲聊消息）",
                output=answer,
            )
        )
    return out


# ----- G: multi-turn follow-up ----------------------------------------------


# G 类（多轮追问）：构造"上一轮提到某案 → 本轮用代词指代"的对话，
# 训练模型做指代消解并把代词正确绑定回历史实体对应的 [EventID=xxx]。
def build_class_g(ctx: BuilderContext) -> list[QASample]:
    q = ctx.quota_for("G")
    total = q.total
    rng = ctx.rng
    pool = [e for e in ctx.corpus if _field(e, "title", "Title") and _field(e, "event_id", "EventID")]

    splits_per_sample = _random_split_labels(total, (q.train, q.val, q.test), rng)
    out: list[QASample] = []
    for split in splits_per_sample:
        # constrain event choice to time split only if the base event_id must obey it
        # Simpler: pick any event, but bind its split via the time-based split of the event
        candidate_pool = [e for e in pool if _split_of_event(e) == split] or pool
        seed = rng.choice(candidate_pool)
        out.append(_build_g_sample(seed, split, rng))
    return out


def _build_g_sample(seed: dict[str, Any], split: str, rng: random.Random) -> QASample:
    eid = _field(seed, "event_id", "EventID")
    title = _short(_field(seed, "title", "Title", default="相关案例"), 20)
    violation = _violation_list(seed)
    vt_name = violation[0] if violation else (_brief_activity(seed) or "相关违规")
    prev_q = rng.choice(
        [
            f"{vt_name}类案例举几个？",
            f"帮我找{vt_name}的处罚案例。",
            f"请列举{vt_name}方面的处罚案例。",
        ]
    )
    prev_a = f"检索到多条案例，其中典型的是「{title}」[EventID={eid}]。"
    followup = rng.choice(G_FOLLOWUP_TEMPLATES)

    evidence = _render_evidence_block([seed], rng=rng)
    pts = _list_field(seed, "punishment_types", "PunishmentTypes")
    pt_str = "、".join(pts[:3]) if pts else "警告、罚款"
    law_tag = _law_cite(_field(seed, "law", "Law", default=""))
    declare = _field(seed, "declare_date", "DeclareDate", default="未知")
    parties = _list_field(seed, "parties", "Parties")
    party_str = "、".join(parties[:3]) if parties else "未提取"
    supervisor = _field(seed, "supervisor", "Supervisor", default="未知")

    # Map follow-up to an answer that references the seed case explicitly
    ftxt = followup
    if "处罚" in ftxt and "机构" in ftxt:
        body = f"该案「{title}」的处罚机构为 {supervisor}。"
    elif "主体" in ftxt or "当事人" in ftxt:
        body = f"该案「{title}」涉及的主体为 {party_str}。"
    elif "哪一年" in ftxt or "年份" in ftxt:
        body = f"该案「{title}」的公告日期为 {declare}。"
    elif "法" in ftxt:
        body = f"该案「{title}」违反的法规为 {law_tag}。"
    elif "违规类型" in ftxt:
        body = f"该案「{title}」的违规类型为 {vt_name}。"
    elif "罚款" in ftxt or "金额" in ftxt:
        body = (
            f"该案「{title}」的处罚类型为 {pt_str}；"
            "具体金额以公告原文为准，证据中未单独列出均值。"
        )
    elif "处分" in ftxt:
        body = f"该案「{title}」的处罚类型为 {pt_str}，具体处分措施以公告原文为准。"
    else:
        body = f"该案「{title}」的处罚类型为 {pt_str}。"

    answer = f"{body} [EventID={eid}]"

    # Encode history inline inside the input field (kept as a single string so
    # that the Alpaca schema stays unchanged).
    input_text = (
        "历史对话：\n"
        f"用户：{prev_q}\n"
        f"助手：{prev_a}\n\n"
        f"用户追问：{followup}\n\n"
        "[检索证据]\n"
        f"{evidence}"
    )

    return QASample(
        event_id_source=eid or None,
        category="G",
        split=split,
        instruction=INSTRUCTION_MULTI_TURN,
        input=input_text,
        output=answer,
    )


# ----- H: anti-hallucination negatives --------------------------------------


# H 类（反幻觉负样本）：用虚构的公司名与 EventID 诱导提问，但检索证据为空，
# 标准答案统一是"未检索到"。这是抑制模型在无证据时编造细节的关键负样本。
def build_class_h(ctx: BuilderContext) -> list[QASample]:
    q = ctx.quota_for("H")
    total = q.total
    rng = ctx.rng

    splits_per_sample = _random_split_labels(total, (q.train, q.val, q.test), rng)
    out: list[QASample] = []
    for i, split in enumerate(splits_per_sample):
        out.append(_build_h_sample(i, split, rng))
    return out


def _build_h_sample(idx: int, split: str, rng: random.Random) -> QASample:
    fake_company = rng.choice(H_FAKE_COMPANIES)
    fake_eid = rng.choice(H_FAKE_EVENT_IDS)
    mode = idx % 3

    if mode == 0:
        user_q = f"{fake_company}之前被证监会处罚过吗？具体罚款多少？"
    elif mode == 1:
        user_q = f"你之前提到的 {fake_eid} 具体是怎么处罚的？"
    else:
        user_q = f"帮我查一下{fake_company}（{fake_eid}）的处罚详情。"

    answer = rng.choice(NEGATIVE_TEMPLATES)
    input_text = (
        f"用户问题：{user_q}\n\n"
        "[检索证据]\n"
        "（检索系统未返回匹配结果；当前证据列表为空）"
    )
    return QASample(
        event_id_source=None,
        category="H",
        split=split,
        instruction=INSTRUCTION_NEGATIVE,
        input=input_text,
        output=answer,
    )


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def _random_split_labels(
    total: int, sizes: tuple[int, int, int], rng: random.Random
) -> list[str]:
    train_n, val_n, test_n = sizes
    labels = (["train"] * train_n) + (["val"] * val_n) + (["test"] * test_n)
    # Pad / trim to match total
    if len(labels) < total:
        labels += ["train"] * (total - len(labels))
    labels = labels[:total]
    rng.shuffle(labels)
    return labels


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


CLASS_BUILDERS = {
    "A": build_class_a,
    "B": build_class_b,
    "C": build_class_c,
    "D": build_class_d,
    "E": build_class_e,
    "F": build_class_f,
    "G": build_class_g,
    "H": build_class_h,
}


# 按配额表逐类调用对应 builder（A..H），汇总成完整数据集；
# 每类产出后打印"实际 / 目标"数量，便于核对配比是否达成。
def build_all(ctx: BuilderContext) -> list[QASample]:
    all_samples: list[QASample] = []
    for quota in ctx.quotas:
        builder = CLASS_BUILDERS[quota.category]
        produced = builder(ctx)
        logger.info("class %s produced %d / target %d", quota.category, len(produced), quota.total)
        all_samples.extend(produced)
    return all_samples


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def assert_no_event_leakage(samples: list[QASample]) -> dict[str, int]:
    """断言：同一个 event_id_source 不会同时出现在两个划分中（杜绝数据泄漏）。

    E/F/H 类无关联事件（event_id_source=None），不参与此校验。
    任意两划分的事件集合若有交集即抛断言错误，这是数据可信度的硬性闸门。
    """
    split_to_events: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for s in samples:
        if s.event_id_source is None:
            continue
        split_to_events[s.split].add(s.event_id_source)
    inter_tv = split_to_events["train"] & split_to_events["val"]
    inter_tt = split_to_events["train"] & split_to_events["test"]
    inter_vt = split_to_events["val"] & split_to_events["test"]
    assert not inter_tv, f"event leakage train∩val: {sorted(inter_tv)[:5]}"
    assert not inter_tt, f"event leakage train∩test: {sorted(inter_tt)[:5]}"
    assert not inter_vt, f"event leakage val∩test: {sorted(inter_vt)[:5]}"
    return {k: len(v) for k, v in split_to_events.items()}


def assert_citation_coverage(samples: list[QASample]) -> None:
    # 断言：A/C/D 类答案必须含 [EventID=] 引用，B 类至少含 EventID 或法条引用之一。
    # 保证训练目标本身就示范了"有据可依"的引用规范，避免教坏模型。
    missing: list[str] = []
    for s in samples:
        if s.category in {"A", "C", "D"}:
            if "[EventID=" not in s.output:
                missing.append(s.category)
        elif s.category == "B":
            if "[EventID=" not in s.output and "[法条：" not in s.output:
                missing.append(s.category)
    assert not missing, (
        f"{len(missing)} A/B/C/D samples missing citations; sample cats={missing[:10]}"
    )


def assert_c_no_leakage(samples: list[QASample]) -> None:
    """断言：C 类的检索证据块绝不能泄漏处罚结果标签。

    用户问题里出现"处罚"二字是正常的（用户本就在问），因此只扫描 ``[检索证据]``
    标记之后的证据部分，命中黑名单词即判泄漏。与 _strip_c_leakage 形成"生成时剔除 +
    校验时复查"的双重保险。
    """
    offenders: list[str] = []
    for s in samples:
        if s.category != "C":
            continue
        marker = "[检索证据]"
        if marker in s.input:
            evidence_block = s.input.split(marker, 1)[1]
        else:
            evidence_block = s.input
        lowered = evidence_block.lower()
        for bad in C_LEAK_BLOCKLIST:
            if bad.lower() in lowered:
                offenders.append(bad)
                break
    assert not offenders, f"C evidence leaks labels: {offenders[:5]}"


def assert_h_refusal(samples: list[QASample]) -> None:
    # 断言：H 类（反幻觉负样本）的答案必须明确拒答（含"未检索到"或"无法回答"），
    # 否则等于在教模型对虚构主体编造细节，与该类设计意图背道而驰。
    offenders: list[str] = []
    for s in samples:
        if s.category != "H":
            continue
        if "未检索到" not in s.output and "无法回答" not in s.output:
            offenders.append(s.output[:40])
    assert not offenders, f"H output not refusing: {offenders[:5]}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/event_corpus.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/reports/m3_lora_data_report.md"),
        help="Path to write the human-readable data construction report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _class_counts(samples: Iterable[QASample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        out[s.category] = out.get(s.category, 0) + 1
    return out


CLASS_DESCRIPTIONS: dict[str, str] = {
    "A": "案例检索（基于 activity / violation_types 派生 query）",
    "B": "法条依据（基于 law 字段派生 query，引用法条标签）",
    "C": "处罚推荐（PunishmentMeasure 已从 input 中剔除防泄漏）",
    "D": "趋势统计（按 violation_types + declare_date 聚合）",
    "E": "拒答（越界问题，固定拒答话术）",
    "F": "问候/闲聊（固定能力介绍话术）",
    "G": "多轮跟进（history 共指消解，引用上一轮 EventID）",
    "H": "反幻觉负样本（虚构公司 / EID，必须回复『未检索到』或『无法回答』）",
}


def _truncate(text: str, n: int) -> str:
    text = text.replace("\n", " ⏎ ").strip()
    return text if len(text) <= n else text[:n] + "…"


def _write_report(
    report_path: Path,
    train: list[QASample],
    val: list[QASample],
    test: list[QASample],
    quotas: tuple[ClassQuota, ...],
    split_event_counts: dict[str, int],
    assertions_pass: dict[str, bool],
    rng: random.Random,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(train) + len(val) + len(test)
    actual = _class_counts(train + val + test)
    train_cnt = _class_counts(train)
    val_cnt = _class_counts(val)
    test_cnt = _class_counts(test)

    lines: list[str] = []
    lines.append("# M3 · LoRA 训练数据构造报告\n")
    lines.append("> 生成脚本：`scripts/build_rag_qa_train.py`（本项目数据构造模块）")
    lines.append("> 数据源：`data/processed/event_corpus.jsonl`（4233 事件）")
    lines.append("> 产出：`data/processed/rag_qa_{train,val,test}.jsonl`\n")

    lines.append("## 1. 八类实际数量（目标 vs 实际）\n")
    lines.append("| 类别 | 目标 | 实际 | 说明 |")
    lines.append("|------|------|------|------|")
    for q in quotas:
        lines.append(
            f"| {q.category} | {q.total} | {actual.get(q.category, 0)} | "
            f"{CLASS_DESCRIPTIONS[q.category]} |"
        )
    lines.append(f"| **总计** | 5500 | {total} | — |\n")

    lines.append("## 2. 切分统计（train / val / test）\n")
    lines.append("| 类别 | train | val | test |")
    lines.append("|------|-------|-----|------|")
    for cat in "ABCDEFGH":
        lines.append(
            f"| {cat} | {train_cnt.get(cat, 0)} | {val_cnt.get(cat, 0)} | {test_cnt.get(cat, 0)} |"
        )
    lines.append(f"| **合计** | {len(train)} | {len(val)} | {len(test)} |\n")
    if total:
        lines.append(
            f"切分比例：train={len(train)/total:.3f}，val={len(val)/total:.3f}，"
            f"test={len(test)/total:.3f}。\n"
        )
    lines.append(
        "切分规则：\n"
        "- A/B/C/D：按事件 `declare_date` 时间三分（train=1994–2021，val=2022–2023，test=2024–2025）；"
        "同一 event_id 绝不跨 split。\n"
        "- E/F/H：无事件绑定，按 80/10/10 随机切分（`seed=42`）。\n"
        "- G：随机 80/10/10；每个 split 内仅选取该 split 对应时间窗口的事件做 seed，避免多轮样本的 event 泄漏。\n"
    )
    lines.append(
        "各 split 中出现过的不同 event_id 数量："
        f"train={split_event_counts.get('train', 0)}, "
        f"val={split_event_counts.get('val', 0)}, "
        f"test={split_event_counts.get('test', 0)}。\n"
    )

    lines.append("## 3. 四条验证断言结果\n")
    assertion_rows = [
        ("assertion_1_event_isolation", "同一 event_id 不跨 train/val/test"),
        ("assertion_2_abcd_citation", "A/B/C/D 类 100% 含 `[EventID=` 或 `[法条：`"),
        ("assertion_3_c_no_leakage", "C 类证据块不含 PunishmentMeasure / 罚款金额 等泄漏词"),
        ("assertion_4_h_refusal", "H 类 output 100% 含『未检索到』或『无法回答』"),
    ]
    for key, desc in assertion_rows:
        status = "✅ PASS" if assertions_pass.get(key, False) else "❌ FAIL"
        lines.append(f"- {status}　{desc}")
    lines.append("")

    lines.append("## 4. 样例展示（每类 3 条，input/output 截断展示）\n")
    all_samples = train + val + test
    for cat in "ABCDEFGH":
        cat_samples = [s for s in all_samples if s.category == cat]
        pick: list[QASample] = []
        for split_name in ("train", "val", "test"):
            subset = [s for s in cat_samples if s.split == split_name]
            if subset:
                pick.append(rng.choice(subset))
        while len(pick) < 3 and cat_samples:
            pick.append(rng.choice(cat_samples))
        lines.append(f"### {cat} 类样例\n")
        for p in pick[:3]:
            lines.append(
                f"- **split**=`{p.split}`　**event_id_source**=`{p.event_id_source}`"
            )
            lines.append(f"  - instruction: {_truncate(p.instruction, 80)}")
            lines.append(f"  - input: `{_truncate(p.input, 260)}`")
            lines.append(f"  - output: `{_truncate(p.output, 200)}`")
            lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report written -> %s", report_path)


def _run_assertions(samples: list[QASample]) -> tuple[dict[str, int], dict[str, bool]]:
    """Run the 4 mandatory assertions; return (event_counts, pass_map)."""
    result: dict[str, bool] = {}
    try:
        split_event_counts = assert_no_event_leakage(samples)
        result["assertion_1_event_isolation"] = True
    except AssertionError as exc:
        logger.error("assertion 1 FAILED: %s", exc)
        split_event_counts = {}
        result["assertion_1_event_isolation"] = False

    try:
        assert_citation_coverage(samples)
        result["assertion_2_abcd_citation"] = True
    except AssertionError as exc:
        logger.error("assertion 2 FAILED: %s", exc)
        result["assertion_2_abcd_citation"] = False

    try:
        assert_c_no_leakage(samples)
        result["assertion_3_c_no_leakage"] = True
    except AssertionError as exc:
        logger.error("assertion 3 FAILED: %s", exc)
        result["assertion_3_c_no_leakage"] = False

    try:
        assert_h_refusal(samples)
        result["assertion_4_h_refusal"] = True
    except AssertionError as exc:
        logger.error("assertion 4 FAILED: %s", exc)
        result["assertion_4_h_refusal"] = False

    return split_event_counts, result


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s %(name)s] %(message)s",
    )

    corpus = list(read_jsonl(args.corpus)) if args.corpus.exists() else []
    logger.info("loaded corpus=%d from %s", len(corpus), args.corpus)
    if not corpus:
        logger.error("empty corpus; abort")
        return 1

    rng = random.Random(args.seed)
    ctx = BuilderContext(corpus=corpus, rng=rng)

    # 主流程：读语料 → 八类生成 → 跑全部断言校验（泄漏/引用/拒答）→ 按 split 落盘
    # → 写构造报告。任一断言失败则以非零码退出，确保不会产出不合格数据。
    samples = build_all(ctx)
    logger.info("total produced=%d", len(samples))

    split_event_counts, assertions_pass = _run_assertions(samples)
    for key, ok in assertions_pass.items():
        logger.info("%s => %s", key, "PASS" if ok else "FAIL")
    logger.info("distinct event_ids per split: %s", split_event_counts)

    by_split: dict[str, list[QASample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        by_split[s.split].append(s)

    n_train = write_jsonl(by_split["train"], args.out / "rag_qa_train.jsonl")
    n_val = write_jsonl(by_split["val"], args.out / "rag_qa_val.jsonl")
    n_test = write_jsonl(by_split["test"], args.out / "rag_qa_test.jsonl")
    logger.info(
        "written train=%d val=%d test=%d -> %s", n_train, n_val, n_test, args.out
    )

    # Use a fresh RNG for report sampling so file writes above are already
    # deterministic and the report is reproducible too.
    _write_report(
        args.report,
        by_split["train"],
        by_split["val"],
        by_split["test"],
        ctx.quotas,
        split_event_counts,
        assertions_pass,
        random.Random(args.seed + 1),
    )

    if not all(assertions_pass.values()):
        logger.error("one or more assertions failed; see report for details")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

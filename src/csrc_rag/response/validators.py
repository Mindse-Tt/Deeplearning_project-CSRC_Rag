"""L7 引证校验层：对生成回答做 8 条规则的后处理与合规检查。

本模块是我们 RAG 流水线的「可信度闸门」，作用于 LLM 或 TemplateResponder 产出的
原始文本，逐条执行以下 8 条校验（部分规则会就地改写文本以自动修复）：

    L7-1  回答中出现的每个 [EventID=xxx] 必须落在本次证据集合内（防伪造引证）。
    L7-2  至少要出现一个 [EventID=]（趋势分析意图豁免，因为它以统计为主）。
    L7-3  [法条：《xx》第xx条] 必须匹配规范正则（结构正确）。
    L7-4  法名 xx 必须落在我们维护的境内证券类法律白名单中（防张冠李戴）。
    L7-5  含数字/金额/百分比的句子，必须同句引用 EventID 或法条；否则记为「无支撑论断」。
    L7-6  全文长度上限 800 字，超出按句界自动截断。
    L7-7  「法院已判决」等越权措辞，替换为带不确定性的中性表述。
    L7-8  处罚推荐意图的输出必须包含免责声明关键词，缺失则自动补上。

模块整体保持纯函数风格：返回一份结构化 ValidationReport，外加一段可能被改写过的
文本（对应 L7-6 / L7-7 / L7-8 三类自动修复）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# 提取案例引证标记 [EventID=xxx]，捕获组为内部的 event_id 原值。
EVENT_ID_RE = re.compile(r"\[EventID=([^\]\s]+?)\]")

# 法条引证正则：兼容中文数字与阿拉伯数字的条号，并支持可选的「之N」后缀（如第二十条之一）。
LAW_CITE_RE = re.compile(
    r"\[法条：《([^》]{1,40})》第([0-9一二三四五六七八九十百零两]+)条(之[0-9一二三四五六七八九十]+)?\]"
)

# 数值型论断识别：百分比 / 金额 / 计数等，用于 L7-5 判定「有数字却无引证」的句子。
NUMERIC_CLAIM_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:%|％|万元|亿元|元|人|年|次|起|件))"
)

# L7-7 越权措辞改写表：把「确定性判决/必须判处」类表述替换为中性、带不确定性的措辞，
# 避免模型对未决案件下达事实性结论而引发合规风险。
FORBIDDEN_PATTERNS = [
    (re.compile(r"法院已判决"), "建议参考"),
    (re.compile(r"根据法律，?必须判处"), "历史案例中常见处罚为"),
    (re.compile(r"本案应判处"), "历史案例中类似情形通常处以"),
]

# L7-8 免责声明关键词：处罚推荐回答必须命中其中之一，否则视为缺失并自动补上。
DISCLAIMER_KEYWORDS = ("仅供参考", "不构成执法", "不构成正式执法")

MAX_ANSWER_CHARS = 800

# L7-4 法名白名单：境内证券/金融监管常用法律，随语料扩充可继续追加。
# 校验时用「包含匹配」，因此「证券法」也能命中「中华人民共和国证券法」等全称。
LAW_WHITELIST = {
    "证券法",
    "公司法",
    "证券投资基金法",
    "基金法",
    "期货和衍生品法",
    "期货法",
    "刑法",
    "反洗钱法",
    "商业银行法",
    "行政处罚法",
    "证券投资者保护法",
    "上市公司信息披露管理办法",
    "证券发行与承销管理办法",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    # 校验报告：记录各条规则的命中明细与整体严重级别，供上层决策是否降级回答/打日志。
    passed: bool = True
    missing_event_ids: list[str] = field(default_factory=list)
    invalid_laws: list[str] = field(default_factory=list)
    unknown_law_names: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    truncated: bool = False
    disclaimer_added: bool = False
    severity: str = "ok"   # ok | low | medium | high | critical

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "missing_event_ids": self.missing_event_ids,
            "invalid_laws": self.invalid_laws,
            "unknown_law_names": self.unknown_law_names,
            "unsupported_claims": self.unsupported_claims,
            "forbidden_hits": self.forbidden_hits,
            "truncated": self.truncated,
            "disclaimer_added": self.disclaimer_added,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    # 校验最终产物：可能被自动修复过的文本 + 合法引证清单 + 详细报告。
    text: str                       # 经自动修复（L7-6/7/8）后的文本
    cited_event_ids: list[str]
    cited_laws: list[str]
    report: ValidationReport


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_event_ids(text: str) -> list[str]:
    # 抽取文中全部 EventID 引证，去重并保持出现顺序。
    seen = set()
    out: list[str] = []
    for m in EVENT_ID_RE.finditer(text):
        eid = m.group(1).strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def extract_law_citations(text: str) -> list[str]:
    """抽取所有符合规范格式的 `[法条：《xx》第xx条]` 引证字符串（去重保序）。"""
    seen = set()
    out: list[str] = []
    for m in LAW_CITE_RE.finditer(text):
        full = m.group(0)
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out


def extract_law_names(text: str) -> list[str]:
    return [m.group(1) for m in LAW_CITE_RE.finditer(text)]


def _split_sentences(text: str) -> list[str]:
    # 按中英文句末标点切句（保留分隔符后的内容），供 L7-5 逐句做引证支撑判定。
    parts = re.split(r"(?<=[。！？!?;；\n])", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------

def check_event_ids(
    text: str, allowed_ids: Iterable[str]
) -> tuple[list[str], list[str]]:
    """L7-1/L7-2 核心：返回 (合法引证, 越界引证)。

    合法 = 落在本次证据集合内；越界 = 回答里出现但证据里没有（即模型伪造的 EventID）。
    """
    allowed = {str(a) for a in allowed_ids}
    cited = extract_event_ids(text)
    valid = [c for c in cited if c in allowed]
    missing = [c for c in cited if c not in allowed]
    return valid, missing


def check_law_format(text: str) -> tuple[list[str], list[str], list[str]]:
    """L7-3/L7-4：返回 (合规法条引证, 残缺引证片段, 不在白名单的法名)。

    「残缺引证片段」指看起来是法条引用（含「法条：」「《》」标记）但未通过严格正则的，
    通常是模型把条号写错或括号缺失，需要标记出来提示人工/降级处理。
    """
    valid = extract_law_citations(text)
    names = extract_law_names(text)
    # 白名单用包含匹配：法名只要包含任一白名单词即视为已知，否则记为未知法名。
    unknown = [n for n in names if not any(w in n for w in LAW_WHITELIST)]
    # 启发式：扫出形如「[法条：...]」但不在合规集合里的片段，即结构破损的引证。
    broken = []
    for raw in re.findall(r"\[法条：[^\]]{0,60}\]", text):
        if raw not in valid:
            broken.append(raw)
    return valid, broken, unknown


def check_unsupported_claims(text: str) -> list[str]:
    """L7-5：找出「含数字但同句无 [EventID=...] 或 [法条：...] 引证」的句子。

    这类句子往往是模型凭空报数（金额、比例、年份），我们把它们记为无支撑论断
    供置信度计算扣分，从源头压制数据幻觉。
    """
    unsupported: list[str] = []
    for sent in _split_sentences(text):
        if NUMERIC_CLAIM_RE.search(sent) and not (
            EVENT_ID_RE.search(sent) or LAW_CITE_RE.search(sent)
        ):
            unsupported.append(sent[:80])
    return unsupported


def apply_forbidden_rewrites(text: str) -> tuple[str, list[str]]:
    # L7-7：按改写表逐条替换越权措辞，返回 (改写后文本, 命中的模式列表)。
    hits: list[str] = []
    out = text
    for pat, repl in FORBIDDEN_PATTERNS:
        if pat.search(out):
            hits.append(pat.pattern)
            out = pat.sub(repl, out)
    return out, hits


def enforce_length(text: str, *, limit: int = MAX_ANSWER_CHARS) -> tuple[str, bool]:
    # L7-6：超长则截断，返回 (文本, 是否被截断)。
    if len(text) <= limit:
        return text, False
    # 在不超过上限的前提下回退到最近的句末标点截断，避免把句子切到一半。
    cut = text[:limit]
    tail = re.search(r"[。！？.!?]", cut[::-1])
    if tail:
        cut = cut[: limit - tail.start()]
    return cut.rstrip() + "……", True


def ensure_disclaimer(text: str) -> tuple[str, bool]:
    # L7-8：处罚推荐回答若缺免责声明，则在末尾自动补一条标准免责语。
    if any(k in text for k in DISCLAIMER_KEYWORDS):
        return text, False
    return text.rstrip() + "\n⚠ 以上内容仅基于历史案例统计，仅供参考，不构成正式执法意见。", True


# ---------------------------------------------------------------------------
# Top-level validator
# ---------------------------------------------------------------------------

def validate(
    text: str,
    *,
    intent_name: str,
    evidence_event_ids: Iterable[str],
) -> ValidationResult:
    """串联执行全部 L7 规则，返回 (可能被改写的文本 + 结构化报告)。

    自动改写按固定顺序进行：
        L7-7 越权措辞替换 → L7-6 长度截断 → L7-8 处罚推荐补免责声明。
    其余规则（L7-1~L7-5）只检测并记录到报告，不改写正文。
    严重级别采用「只升不降」策略：任一规则命中都把 severity 抬到更高档。
    """
    report = ValidationReport()

    # L7-7：越权措辞替换，对所有意图始终执行；命中则抬到 high 级。
    text, forbidden_hits = apply_forbidden_rewrites(text)
    if forbidden_hits:
        report.forbidden_hits = forbidden_hits
        report.severity = _bump(report.severity, "high")

    # L7-1 / L7-2：EventID 引证检查。
    valid_ids, missing_ids = check_event_ids(text, evidence_event_ids)
    if missing_ids:
        # 伪造引证是最严重问题——直接判不通过并标 critical。
        report.missing_event_ids = missing_ids
        report.passed = False
        report.severity = _bump(report.severity, "critical")
    if not valid_ids and intent_name != "trend_analysis":
        # 除趋势分析外，回答里一个合法引证都没有同样判不通过（high）。
        report.passed = False
        report.severity = _bump(report.severity, "high")

    # L7-3 / L7-4：法条格式 + 白名单检查。
    valid_laws, invalid_laws, unknown_names = check_law_format(text)
    if invalid_laws:
        report.invalid_laws = invalid_laws
        report.passed = False
        report.severity = _bump(report.severity, "high")
    if unknown_names:
        report.unknown_law_names = unknown_names
        report.severity = _bump(report.severity, "medium")

    # L7-5：数值论断缺引证，记入报告并抬到 medium（不强制判不通过，交置信度扣分）。
    unsupported = check_unsupported_claims(text)
    if unsupported:
        report.unsupported_claims = unsupported
        report.severity = _bump(report.severity, "medium")

    # L7-6：长度截断（改写正文）。
    text, truncated = enforce_length(text)
    report.truncated = truncated

    # L7-8：仅处罚推荐意图需要补免责声明（改写正文）。
    if intent_name == "sanction_recommendation":
        text, added = ensure_disclaimer(text)
        report.disclaimer_added = added

    return ValidationResult(
        text=text,
        cited_event_ids=valid_ids,
        cited_laws=valid_laws,
        report=report,
    )


# ---------------------------------------------------------------------------
# Confidence scoring (§6 of the strategy doc)
# ---------------------------------------------------------------------------

def compute_confidence(
    *,
    avg_top3_score: float,
    answer_text: str,
    n_evidence: int,
    report: ValidationReport,
) -> float:
    """计算回答置信度，取值 ∈ [0, 1]。

    四项加权组合：检索质量(0.4) + 引证覆盖率(0.3) + 证据充分度(0.2) + 无幻觉度(0.1)；
    若存在伪造引证（missing_event_ids），最终分再乘 0.5 作为重罚。
    """
    # 引证覆盖率：含引证的句子数 / 总句数，粗略衡量「每句话是否有据可依」。
    sentences = _split_sentences(answer_text)
    if not sentences:
        citation_coverage = 0.0
    else:
        cited_sents = sum(
            1 for s in sentences if EVENT_ID_RE.search(s) or LAW_CITE_RE.search(s)
        )
        citation_coverage = cited_sents / len(sentences)

    # 证据充分度：召回条数归一到 [0,1]，5 条及以上即视为充分。
    evidence_norm = min(n_evidence, 5) / 5.0

    # 无幻觉度的反向量：无支撑论断句占比，越高说明幻觉越多。
    total_claims = max(1, len(_split_sentences(answer_text)))
    unsupported_ratio = len(report.unsupported_claims) / total_claims

    score = (
        0.4 * max(0.0, min(1.0, avg_top3_score))
        + 0.3 * citation_coverage
        + 0.2 * evidence_norm
        + 0.1 * (1.0 - unsupported_ratio)
    )
    if report.missing_event_ids:
        score *= 0.5  # 出现伪造引证时强罚减半，置信度必须显著下降
    return max(0.0, min(1.0, round(score, 4)))


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

# 严重级别从低到高的有序枚举，_bump 据此实现「只升不降」。
_SEVERITY_ORDER = ["ok", "low", "medium", "high", "critical"]


def _bump(current: str, candidate: str) -> str:
    # 取当前级别与候选级别中更高的一档；遇到未知级别则直接采用候选值兜底。
    try:
        return _SEVERITY_ORDER[max(
            _SEVERITY_ORDER.index(current), _SEVERITY_ORDER.index(candidate)
        )]
    except ValueError:
        return candidate


__all__ = [
    "EVENT_ID_RE",
    "LAW_CITE_RE",
    "LAW_WHITELIST",
    "ValidationReport",
    "ValidationResult",
    "extract_event_ids",
    "extract_law_citations",
    "check_event_ids",
    "check_law_format",
    "check_unsupported_claims",
    "apply_forbidden_rewrites",
    "enforce_length",
    "ensure_disclaimer",
    "validate",
    "compute_confidence",
]

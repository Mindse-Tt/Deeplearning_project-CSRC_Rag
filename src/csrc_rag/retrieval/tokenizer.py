"""L3 中文分词：BM25 / 稀疏检索的分词前处理（本项目检索质量的地基之一）。

中文没有天然词边界，分词质量直接决定 BM25 召回上限。我们为此设计了双后端，
通过 ``configs/retrieval.json::tokenizer`` 切换：

* ``"jieba"``         —— 生产主用。jieba 精确模式分词 + 停用词裁剪，并把
                         synonyms.json 里的领域规范词作为 user_dict 注入，
                         保证 "内幕交易""操纵市场""证券法" 这类专业术语不被
                         过度切碎、作为单一 token 参与 BM25 打分。
* ``"regex_bigram"``  —— 消融/兜底基线。ALNUM + 书名号法规名 + 中文滑动
                         bigram 的纯规则切分，保留它是为了能稳定复现 M2 阶段
                         基线，也作为 jieba 不可用时的降级路径。

后端在每个进程内只解析一次 retrieval.json，调用方无需关心；下游索引
（``BM25Index``）只需以纯字符串调用 :func:`tokenize`。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared regex for the legacy (regex_bigram) backend
# ---------------------------------------------------------------------------

ALNUM_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
LAW_PATTERN = re.compile(r"《[^》]{1,40}》")


# ---------------------------------------------------------------------------
# Inline Chinese + English stop-word list (kept small and domain-aware).
#
# Notes
# -----
# * We intentionally keep domain terms like "公司" / "股份" OUT of the list
#   because they carry signal in our corpus (firm-name matching).
# * Punctuation and single-character helpers dominate; this is enough to
#   recover BM25 discrimination without dragging in a full HIT list.
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        # Chinese function words
        "的", "了", "和", "与", "及", "或", "而", "及其", "等", "是", "在", "于",
        "对", "对于", "以", "被", "把", "将", "从", "从而", "因", "因此",
        "所以", "并", "并且", "也", "还", "又", "就", "都", "不", "没", "没有",
        "有", "这", "那", "这个", "那个", "这些", "那些", "之", "其", "其中",
        "如", "如同", "即", "乃", "但", "但是", "然而", "则", "向", "到",
        "为", "给", "让", "使", "令", "着", "过", "了",
        "吧", "吗", "呢", "啊", "呀", "哦", "嗯", "哈", "哼",
        "已", "已经", "正在", "正", "尚", "曾",
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
        "请问", "请", "帮", "帮我", "告诉", "说明", "介绍",
        "怎么", "如何", "什么", "哪些", "哪个", "多少", "几", "若干",
        "可", "可以", "能", "能够", "需要", "需", "应", "应当", "必须",
        "的话", "来", "去", "出", "上", "下", "里", "内", "外",
        # Punctuation often leaks through
        "。", "，", "、", "；", "：", "？", "！", "“", "”", "‘", "’",
        "（", "）", "【", "】", "《", "》", "—", "…", "·",
        ".", ",", ";", ":", "?", "!", "\"", "'",
        "(", ")", "[", "]", "-", "—", "…",
        # English function words (robustness)
        "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "at",
        "by", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its",
    }
)


def _project_root() -> Path:
    # src/csrc_rag/retrieval/tokenizer.py -> project root is 3 levels up
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Backend state (lazy-loaded, per-process)
# ---------------------------------------------------------------------------

_BACKEND: str | None = None
_JIEBA_READY: bool = False
_JIEBA_MODULE = None


def _load_backend_from_config() -> str:
    """Read ``tokenizer`` field from configs/retrieval.json. Default 'regex_bigram'."""
    cfg_path = _project_root() / "configs" / "retrieval.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        LOGGER.warning("retrieval.json not found at %s; defaulting to regex_bigram", cfg_path)
        return "regex_bigram"
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to parse retrieval.json (%s); defaulting to regex_bigram", exc)
        return "regex_bigram"
    backend = str(cfg.get("tokenizer", "regex_bigram")).lower()
    if backend not in {"jieba", "regex_bigram"}:
        LOGGER.warning("Unknown tokenizer=%s; defaulting to regex_bigram", backend)
        return "regex_bigram"
    return backend


def _canonical_terms_from_synonyms() -> list[str]:
    """Collect canonical terms from configs/synonyms.json for jieba user_dict.

    Both the categorised (``raw[category][canonical] = [aliases]``) and the
    legacy flat (``raw['synonyms'][canonical] = [aliases]``) layouts are
    supported, matching the tolerance already in slot_filler.
    """
    path = _project_root() / "configs" / "synonyms.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("tokenizer: failed to read synonyms.json: %s", exc)
        return []

    canonicals: set[str] = set()
    if isinstance(raw, dict):
        # Categorised layout
        for key, block in raw.items():
            if key.startswith("_") or not isinstance(block, dict):
                continue
            if key == "synonyms":
                # Flat layout lives under this key
                for canon, aliases in block.items():
                    canonicals.add(canon)
                    if isinstance(aliases, list):
                        for a in aliases:
                            if isinstance(a, str):
                                canonicals.add(a)
                continue
            # Categorised: e.g. raw["violation"]["内幕交易"] = [aliases]
            for canon, aliases in block.items():
                if not isinstance(canon, str):
                    continue
                canonicals.add(canon)
                if isinstance(aliases, list):
                    for a in aliases:
                        if isinstance(a, str):
                            canonicals.add(a)
        # violation_types list (legacy)
        vio_list = raw.get("violation_types")
        if isinstance(vio_list, list):
            for v in vio_list:
                if isinstance(v, str):
                    canonicals.add(v)
    return sorted(canonicals, key=len, reverse=True)


def _init_jieba() -> bool:
    """Initialise jieba + inject canonicals as user_dict. Returns False on failure."""
    global _JIEBA_READY, _JIEBA_MODULE
    if _JIEBA_READY:
        return True
    try:
        import jieba  # type: ignore
    except ImportError:
        LOGGER.warning(
            "jieba not installed; BM25 tokenizer will fall back to regex_bigram. "
            "Run `pip install jieba` to enable."
        )
        return False
    # 把同义词表里的领域规范词注入 jieba 自定义词典，避免多字术语被切碎。
    # 这是本项目让 BM25 在金融监管语料上"对得上术语"的关键一步：例如不注入时
    # "内幕交易" 可能被切成 "内幕"+"交易"，丢失专有词的判别力。
    for term in _canonical_terms_from_synonyms():
        if len(term) >= 2 and len(term) <= 20:
            # freq=1000 给一个足够高的词频权重，压过 jieba 默认词典里的拆分倾向；
            # 不指定词性（POS），因为下游只用 token 字面、不依赖词性标注。
            jieba.add_word(term, freq=1000)
    # Silence jieba's initial "Building prefix dict from the default dictionary" log.
    jieba.initialize()
    _JIEBA_MODULE = jieba
    _JIEBA_READY = True
    return True


def _get_backend() -> str:
    """Return the resolved backend, falling back if jieba is unavailable."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    backend = _load_backend_from_config()
    if backend == "jieba" and not _init_jieba():
        backend = "regex_bigram"
    _BACKEND = backend
    LOGGER.info("tokenizer backend resolved to: %s", backend)
    return backend


def reset_backend_cache() -> None:
    """Reset the resolved backend (test hook / ablation harness)."""
    global _BACKEND
    _BACKEND = None


# ---------------------------------------------------------------------------
# Legacy regex + bigram backend (kept byte-identical for ablation)
# ---------------------------------------------------------------------------


def _tokenize_regex_bigram(text: str) -> list[str]:
    normalized = text.lower().strip()
    tokens: list[str] = []

    for law_name in LAW_PATTERN.findall(normalized):
        tokens.append(law_name)

    for token in ALNUM_PATTERN.findall(normalized):
        tokens.append(token)

    for chunk in CJK_PATTERN.findall(normalized):
        compact = chunk.strip()
        if not compact:
            continue
        if len(compact) <= 2:
            tokens.append(compact)
            continue
        tokens.append(compact[:6])
        for idx in range(len(compact) - 1):
            tokens.append(compact[idx : idx + 2])

    return tokens


# ---------------------------------------------------------------------------
# Jieba backend
# ---------------------------------------------------------------------------


def _tokenize_jieba(text: str) -> list[str]:
    """jieba precise-mode + stop-word filter + law-name preservation.

    Preservation steps
    ------------------
    1. Pull out ``《...》`` spans first and emit them as single tokens --
       jieba would split these otherwise.
    2. Lower-case and feed the remainder through ``jieba.lcut`` precise
       mode (default).
    3. Drop pure whitespace, stop-words, and 1-char CJK pieces unless they
       are alnum (we keep 1-char English/number tokens like 'a' filtered
       out via stop-words too).
    """
    normalized = text.lower().strip()
    if not normalized:
        return []

    tokens: list[str] = []

    # 步骤 1：先把《...》整段法规名抽出来作为单 token——jieba 会把书名号拆开，
    # 而 "《证券法》" 整体才是检索时最有判别力的实体，必须保全。
    masked = normalized
    for law_name in LAW_PATTERN.findall(normalized):
        tokens.append(law_name)
        # 用等长空格遮盖已抽取的法规名，保证剩余文本的字符位置不偏移，
        # 后续 jieba 切分不会受影响（避免重复切出法规名内部碎片）。
        masked = masked.replace(law_name, " " * len(law_name), 1)

    # 步骤 2：对遮盖后的文本做 jieba 精确模式切分。
    assert _JIEBA_MODULE is not None  # backend init guarantees this
    for seg in _JIEBA_MODULE.lcut(masked, cut_all=False, HMM=True):
        tok = seg.strip()
        if not tok:
            continue
        if tok in _STOPWORDS:
            continue
        # \u4e22\u5f03\u5355\u5b57\u4e2d\u6587\u586b\u5145\u8bcd\uff1a\u5355\u5b57\u4e2d\u6587\u901a\u5e38\u65e0\u5224\u522b\u529b\u4e14\u4f1a\u62ac\u9ad8\u566a\u58f0 TF\uff0c
        # \u4f46\u4fdd\u7559\u5355\u5b57\u82f1\u6587/\u6570\u5b57\uff08\u5982\u4ee3\u7801\u6807\u8bc6\u3001\u6848\u53f7\u7247\u6bb5\uff09\u4ee5\u514d\u8bef\u4f24\u3002
        if len(tok) == 1 and CJK_PATTERN.fullmatch(tok):
            continue
        # \u515c\u5e95\u6e05\u9664\u6f0f\u7f51\u7684\u7eaf\u6807\u70b9 token\uff08\u505c\u7528\u8bcd\u8868\u96be\u4ee5\u7a77\u4e3e\u6240\u6709\u6807\u70b9\u7ec4\u5408\uff09\u3002
        if not re.search(r"[\w\u4e00-\u9fff]", tok):
            continue
        tokens.append(tok)

    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tokenize(text: str | None) -> list[str]:
    """Tokenise ``text`` using the backend configured in retrieval.json."""
    if not text:
        return []

    backend = _get_backend()
    if backend == "jieba":
        return _tokenize_jieba(text)
    return _tokenize_regex_bigram(text)

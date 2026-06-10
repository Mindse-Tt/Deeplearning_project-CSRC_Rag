"""Augment Planner v2 intent seeds (21 rows) to a 3500-row training set.

Strategy (no API): 25 rich skeleton templates per class with slot placeholders
(``{company}``, ``{violation}``, ``{year}``, ``{law}``, ``{agency}`` ...).
Each template is expanded into 20 concrete variants drawn deterministically
from the variable pools below.  25 * 20 = 500 rows/class * 7 classes = 3500.

Variable pools:

* ``companies``  - top 100 high-frequency listed entities pulled from
  ``data/processed/event_corpus.jsonl`` (filtered to公司/集团/证券/控股, drops
  会计所/律所/评估机构 which dominate raw counts).
* ``violations`` - 6 canonical CSRC violation families.
* ``years``      - 1994-2025 (covers the CSRC electronic records range).
* ``laws``       - 10 frequently cited regulatory instruments.
* ``agencies``   - 8 CSRC HQ + local bureaus + exchanges.

Output format (JSONL, one record per line)::

    {"text": "...", "label": "case_retrieval"}

Output path: ``data/processed/intent_train_v2.jsonl`` (3500 rows).

This script is deterministic (``random.seed(42)``) so re-runs produce identical
training corpora, which keeps the model's macro-F1 reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csrc_rag.settings import CONFIG_DIR, PROCESSED_DIR  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SEED_PATH = CONFIG_DIR / "intent_examples_v2.json"
DEFAULT_CORPUS_PATH = PROCESSED_DIR / "event_corpus.jsonl"
DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "intent_train_v2.jsonl"

RANDOM_SEED = 42
ROWS_PER_CLASS = 500
TEMPLATES_PER_CLASS = 25
VARIANTS_PER_TEMPLATE = 20  # 25 * 20 = 500

# --------------------------------------------------------------------------- #
# Variable pools                                                              #
# --------------------------------------------------------------------------- #

VIOLATIONS: tuple[str, ...] = (
    "信息披露违规",
    "内幕交易",
    "市场操纵",
    "虚假陈述",
    "短线交易",
    "违规买卖股票",
)

YEARS: tuple[str, ...] = tuple(str(y) for y in range(1994, 2026))

LAWS: tuple[str, ...] = (
    "《证券法》",
    "《证券投资基金法》",
    "《上市公司信息披露管理办法》",
    "《上市公司收购管理办法》",
    "《刑法》",
    "《证券发行与承销管理办法》",
    "《证券市场禁入规定》",
    "《行政处罚法》",
    "《公司法》",
    "《上市公司治理准则》",
)

AGENCIES: tuple[str, ...] = (
    "证监会",
    "中国证监会",
    "上海证券交易所",
    "深圳证券交易所",
    "北京证券交易所",
    "上海证监局",
    "广东证监局",
    "江苏证监局",
)

# Small curated fallback in case event_corpus is unavailable.
# Also injected alongside corpus-mined names so famous fraud cases (康美/康得新/
# 乐视/瑞幸 ...) are always represented even when absent from our snapshot.
FALLBACK_COMPANIES: tuple[str, ...] = (
    "上海中毅达股份有限公司",
    "海通证券股份有限公司",
    "杭州天目山药业股份有限公司",
    "深圳大通实业股份有限公司",
    "广西慧球科技股份有限公司",
    "中信证券股份有限公司",
    "康得新复合材料集团股份有限公司",
    "国信证券股份有限公司",
    "广发证券股份有限公司",
    "方正证券股份有限公司",
    "兴业证券股份有限公司",
    "平安证券有限责任公司",
    "上海电气集团股份有限公司",
    "北大方正集团有限公司",
    "山东墨龙石油机械股份有限公司",
    "沈阳机床(集团)有限责任公司",
    "辅仁药业集团制药股份有限公司",
    "广东紫晶信息存储技术股份有限公司",
    "泽达易盛(天津)科技股份有限公司",
    "丹东欣泰电气股份有限公司",
)

# Well-known violation cases - always included so they appear in training data
# regardless of corpus coverage. Short forms are intentional to match typical
# colloquial queries (e.g., "康美药业", not "康美药业股份有限公司").
FAMOUS_CASE_COMPANIES: tuple[str, ...] = (
    "康美药业",
    "康美药业股份有限公司",
    "康得新",
    "康得新复合材料集团股份有限公司",
    "乐视网",
    "乐视网信息技术股份有限公司",
    "瑞幸咖啡",
    "ST 康美",
    "獐子岛",
    "獐子岛集团股份有限公司",
    "欣泰电气",
    "丹东欣泰电气股份有限公司",
    "雅百特",
    "江苏雅百特科技股份有限公司",
    "金亚科技",
    "尔康制药",
    "千山药机",
    "中安消",
    "长生生物",
    "天目山药业",
    "抚顺特钢",
    "华泽钴镍",
    "慧球科技",
    "大智慧",
    "蓝田股份",
    "银广夏",
    "万福生科",
    "绿大地",
    "紫鑫药业",
    "海联讯",
)

# --------------------------------------------------------------------------- #
# Company extractor                                                            #
# --------------------------------------------------------------------------- #

_COMPANY_KEEP = re.compile(r"(公司|集团|银行|证券|基金|保险|股份|控股)")
_COMPANY_DROP = re.compile(r"(事务所|特殊普通合伙|评估|律师)")


def load_top_companies(corpus_path: Path, limit: int = 100) -> list[str]:
    """Return up to ``limit`` high-frequency listed entities from the corpus.

    Falls back to ``FALLBACK_COMPANIES`` if the corpus file is missing or empty.
    """
    if not corpus_path.exists():
        logger.warning("Corpus missing (%s); using fallback company pool.", corpus_path)
        return list(FALLBACK_COMPANIES)

    counter: Counter[str] = Counter()
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            parties = record.get("parties")
            if not isinstance(parties, list):
                continue
            for party in parties:
                if not isinstance(party, str):
                    continue
                name = party.strip()
                if not name:
                    continue
                counter[name] += 1

    companies: list[str] = []
    for name, _count in counter.most_common():
        if _COMPANY_KEEP.search(name) and not _COMPANY_DROP.search(name):
            companies.append(name)
        if len(companies) >= limit:
            break

    if not companies:
        return list(FALLBACK_COMPANIES)

    # Always prepend famous violation cases so models trained on this set
    # recognise colloquial queries like "康美药业被罚过什么".
    for name in FAMOUS_CASE_COMPANIES:
        if name not in companies:
            companies.insert(0, name)
    return companies[: max(limit, len(FAMOUS_CASE_COMPANIES))]


# --------------------------------------------------------------------------- #
# Skeleton templates - 25 per class                                            #
# --------------------------------------------------------------------------- #

GREETING_TEMPLATES: tuple[str, ...] = (
    "{greet}",
    "{greet}，{tail}",
    "{greet}{punct}",
    "{greet}，在吗？",
    "{greet}，在线吗？",
    "{greet}，能帮我一下吗",
    "{greet}，麻烦问一下",
    "{greet}，{honorific}",
    "{greet}，我想问点事情",
    "{greet}，听得见吗",
    "{greet}，开始了吗",
    "{greet}，有人在吗",
    "{greet}，今天在忙吗",
    "{greet}，请多关照",
    "{greet}，有空聊两句吗",
    "{greet}，请问可以咨询吗",
    "{greet}，不好意思打扰了",
    "{greet}啊",
    "{greet}呀",
    "{greet}{honorific}你好",
    "{honorific}，{greet}",
    "{greet}，我来了",
    "{greet}，请问能开始工作吗",
    "{greet}，看到请回复一下",
    "{greet}，现在方便吗",
)

GREETING_VARS = {
    "greet": (
        "你好", "您好", "嗨", "hi", "Hi", "hello", "Hello", "早上好",
        "中午好", "下午好", "晚上好", "早安", "午安", "晚安",
        "早", "Hey", "hey", "哈喽", "哈啰", "嘿",
    ),
    "tail": (
        "请多关照", "麻烦你了", "有点事咨询", "今天也辛苦了",
        "在吗", "我来啦", "好久不见", "刚开电脑",
        "先打个招呼", "我是新用户", "想试试你", "在上班吗",
        "顺便问下", "听得见吗", "现在有空吗", "能说话吗",
        "麻烦一下", "开工", "上线啦", "早点休息",
    ),
    "punct": (
        "！", "~", "。", "", "?", "!", " ", "😊",
        "😀", "……", "～", "!!", "？？", ".", "，",
        "呀", "哦", "嘛", "呢", "咯",
    ),
    "honorific": (
        "老师", "小助手", "助理", "AI", "Claude", "机器人", "同学",
        "朋友", "先生", "女士", "大佬", "大神", "师傅", "老板",
        "同事", "您", "亲", "伙计", "童鞋", "哥们",
    ),
}

CHITCHAT_TEMPLATES: tuple[str, ...] = (
    "你叫什么名字",
    "你是谁",
    "你是哪家公司开发的",
    "你是什么模型",
    "你能做什么",
    "你会些什么",
    "你都有哪些功能",
    "介绍一下你自己",
    "你多大了",
    "你是人工智能吗",
    "你是 GPT 吗",
    "你有情感吗",
    "你累不累",
    "你喜欢什么",
    "你会说英语吗",
    "{tone}，和你聊聊天",
    "{tone}，陪我说说话",
    "我有点无聊，{chat_action}",
    "讲个笑话听听",
    "给我讲个故事吧",
    "你平时都干什么",
    "你知道自己是 AI 吗",
    "你觉得自己聪明吗",
    "你的训练数据截止到哪一年",
    "你能自我介绍一下吗",
)

CHITCHAT_VARS = {
    "tone": (
        "你好呀", "嘿", "hi", "嗨", "嘿嘿", "哈喽", "hello", "Hey",
        "喂喂", "在吗", "你好", "您好", "早上好", "晚上好",
        "中午好", "下午好", "我来啦", "闲着没事", "随便聊聊", "开始吧",
    ),
    "chat_action": (
        "陪我聊聊", "和我说说话", "说点什么", "来段有趣的",
        "讲讲冷知识", "聊聊你自己", "随便说点什么", "讲个段子",
        "说说你平时干啥", "讲讲最近的新鲜事", "来个脑筋急转弯",
        "给我点灵感", "分享一件趣事", "出个谜语", "讲点有趣的",
        "来一句名言", "讲讲你的梦想", "说说你的爱好", "推荐点话题",
        "随便聊几句",
    ),
}

OUT_OF_SCOPE_TEMPLATES: tuple[str, ...] = (
    "今天{city}天气怎么样",
    "明天{city}会下雨吗",
    "帮我写一首关于{topic}的诗",
    "写一段关于{topic}的散文",
    "推荐一部{genre}电影",
    "推荐一本{genre}小说",
    "帮我写一段 {lang} 代码实现{algo}",
    "给我讲一下{algo}算法",
    "帮我规划一下去{city}的旅行",
    "{city}有什么好吃的",
    "{city}有哪些景点值得去",
    "帮我算一下{math}",
    "帮我做一道{subject}作业题",
    "{horoscope}今天运势如何",
    "{celeb}最近有什么新闻",
    "{sport}比赛结果怎么样",
    "{crypto}今天涨了吗",
    "帮我写一封{topic}的情书",
    "给我的简历提点建议",
    "帮我翻译一段{lang}文字",
    "{genre}游戏有什么推荐",
    "解梦：我梦见了{dream_thing}",
    "给我推荐一款{product}",
    "{city}最近有什么演出",
    "教我做一道{food}",
)

OOS_VARS = {
    "city": (
        "北京", "上海", "广州", "深圳", "杭州", "成都", "西安",
        "武汉", "南京", "重庆", "长沙", "苏州", "天津", "厦门",
        "青岛", "大连", "昆明", "三亚", "拉萨", "哈尔滨",
    ),
    "topic": (
        "春天", "大海", "爱情", "乡愁", "母亲", "友谊", "青春",
        "秋天", "梦想", "离别", "月亮", "故乡", "童年", "星空",
        "远方", "时间", "自由", "孤独", "勇气", "城市",
    ),
    "genre": (
        "科幻", "悬疑", "爱情", "喜剧", "动作", "恐怖", "文艺",
        "历史", "奇幻", "冒险", "战争", "犯罪", "动画", "纪录",
        "家庭", "灾难", "传记", "音乐", "青春", "武侠",
    ),
    "lang": (
        "Python", "Java", "Go", "Rust", "C++", "JavaScript", "TypeScript",
        "Kotlin", "Swift", "PHP", "Ruby", "Scala", "Lua", "Bash",
        "SQL", "中文", "英文", "日文", "法文", "德文",
    ),
    "algo": (
        "冒泡排序", "快速排序", "归并排序", "二分查找", "深度优先搜索",
        "广度优先搜索", "动态规划", "最短路径", "背包问题", "KMP",
        "哈希表", "红黑树", "堆排序", "贪心算法", "回溯算法",
        "并查集", "拓扑排序", "LRU 缓存", "斐波那契", "最小生成树",
    ),
    "math": (
        "3+5*2", "sqrt(144)", "100 的 15% 是多少", "2 的 10 次方",
        "1 到 100 的和", "圆周率前 5 位", "log2(1024)", "sin(30°)",
        "cos(60°)", "5 的阶乘", "7+8", "9*9", "123/7",
        "45 开平方", "黄金分割比", "e 的值", "π 的值",
        "极限 sin(x)/x", "导数 x^2", "积分 x",
    ),
    "subject": (
        "物理", "化学", "数学", "英语", "语文", "生物", "历史",
        "地理", "政治", "音乐", "美术", "体育", "计算机",
        "经济学", "哲学", "心理学", "法学", "社会学", "统计学", "会计学",
    ),
    "horoscope": (
        "白羊座", "金牛座", "双子座", "巨蟹座", "狮子座", "处女座",
        "天秤座", "天蝎座", "射手座", "摩羯座", "水瓶座", "双鱼座",
        "属鼠", "属牛", "属虎", "属兔", "属龙", "属蛇", "属马", "属羊",
    ),
    "celeb": (
        "刘德华", "周杰伦", "王菲", "马斯克", "成龙", "章子怡",
        "易烊千玺", "杨幂", "蔡徐坤", "周润发", "巩俐", "赵本山",
        "郭德纲", "李连杰", "Taylor Swift", "C 罗", "梅西",
        "姚明", "林书豪", "孙杨",
    ),
    "sport": (
        "NBA", "英超", "西甲", "中超", "意甲", "德甲", "欧冠", "世界杯",
        "奥运会", "亚运会", "CBA", "网球公开赛", "高尔夫大师赛",
        "F1", "环法", "冬奥会", "全运会", "乒乓球世锦赛",
        "羽毛球世锦赛", "MLB",
    ),
    "crypto": (
        "比特币", "以太坊", "狗狗币", "莱特币", "USDT", "XRP", "BNB",
        "SOL", "ADA", "DOT", "TRX", "AVAX", "ATOM", "LTC",
        "DOGE", "SHIB", "MATIC", "LINK", "FIL", "NEAR",
    ),
    "dream_thing": (
        "蛇", "水", "飞翔", "掉牙", "前任", "死去的亲人", "考试",
        "迷路", "被追赶", "地震", "火", "猫", "狗", "婴儿",
        "结婚", "坟墓", "血", "天使", "怪物", "楼梯",
    ),
    "product": (
        "手机", "耳机", "笔记本", "相机", "平板", "智能手表", "吸尘器",
        "空气净化器", "咖啡机", "扫地机器人", "电饭煲", "跑鞋",
        "背包", "键盘", "鼠标", "显示器", "音响", "台灯",
        "护肤品", "洗衣机",
    ),
    "food": (
        "番茄炒蛋", "宫保鸡丁", "鱼香肉丝", "麻婆豆腐", "红烧肉",
        "糖醋里脊", "水煮鱼", "东坡肉", "蛋炒饭", "酸辣土豆丝",
        "意大利面", "咖喱鸡", "寿司", "牛排", "三明治",
        "烤鸭", "饺子", "包子", "粥", "凉拌黄瓜",
    ),
}

CASE_RETRIEVAL_TEMPLATES: tuple[str, ...] = (
    "帮我找和{violation}类似的处罚案例",
    "历史上有没有和{violation}相近的案件",
    "召回和{violation}接近的历史案例",
    "{company}被罚过{violation}吗",
    "{company}被罚过什么",
    "{company}被处罚过吗",
    "{company}有没有被证监会处罚过",
    "{company}因什么被罚",
    "{company}有没有因{violation}被处罚的记录",
    "查一下{company}的历史处罚案件",
    "{company}涉及过哪些违规",
    "列几个{violation}的典型案例",
    "给我几个{violation}的处罚先例",
    "有哪些公司被{agency}因{violation}处罚过",
    "近{year}年有哪些{violation}的案件",
    "{year}年{violation}的典型案例有哪些",
    "哪些上市公司有过{violation}的案底",
    "帮我检索一下和{company}情况类似的案件",
    "与{violation}接近的历史处罚能否召回几条",
    "{agency}处理过哪些{violation}的案件",
    "有没有和这种违规行为类似的先例",
    "帮我找几个证券{violation}的代表性案件",
    "{company}是否有{violation}的违规记录",
    "哪些案件同时涉及{violation}和信息披露违规",
    "{year}年{company}有没有被处罚",
)

LAW_GROUNDING_TEMPLATES: tuple[str, ...] = (
    "{violation}通常违反{law}的哪些条款",
    "{violation}的法律依据是什么",
    "请给出{violation}相关的法律条文",
    "{violation}对应{law}哪一条",
    "{company}的{violation}行为适用哪条法律",
    "关于{violation}，{law}有哪些规定",
    "引用{law}说明{violation}的违法性",
    "请列出处理{violation}可援引的法条",
    "{violation}在{law}中属于哪一类违法行为",
    "{law}第几条涉及{violation}",
    "帮我查一下{violation}的法律依据",
    "能否说明{violation}违反的是{law}还是《刑法》",
    "{agency}在处罚{violation}时一般援引哪些法条",
    "从法律层面讲，{violation}的定性依据是什么",
    "请给出{violation}在{law}中的具体法条编号",
    "想知道{violation}涉及的主要法律规范",
    "{violation}违反了哪些法律和部门规章",
    "这种{violation}行为依据哪条法规处罚",
    "{violation}行为的法律定性请给出条文",
    "请把{violation}涉及的所有法律依据列一下",
    "想核对一下{violation}的法律条款清单",
    "{law}中关于{violation}的规定是第几条",
    "请援引{law}说明{violation}的违法性",
    "涉及{violation}的相关法律条款有哪些",
    "请说明{violation}行为的法律依据",
)

SANCTION_TEMPLATES: tuple[str, ...] = (
    "结合案例给出{violation}的处罚建议",
    "{company}{violation}的处罚建议怎么给",
    "这种{violation}情况更可能警告还是罚款",
    "预测一下{violation}可能对应哪些处罚",
    "{violation}一般会被处以多少罚款",
    "{violation}是否会采取市场禁入措施",
    "请结合处罚先例给出量罚建议",
    "这种{violation}违规应该如何处罚",
    "{company}的案件应如何量罚",
    "综合考虑，{violation}的处罚力度怎么建议",
    "针对{violation}的情节，给出处罚决定书建议",
    "{violation}涉及金额较大，处罚会加重吗",
    "是否会对{company}采取行政处罚",
    "建议对{violation}采取哪些行政措施",
    "帮我给出{violation}的裁量建议",
    "这个{violation}案件适合警告还是罚没",
    "请推荐一个合适的处罚方案",
    "按以往案例，{violation}处罚金额的区间是多少",
    "{violation}是否应终身禁入证券市场",
    "对{company}{violation}行为，给一个量罚建议",
    "{agency}通常对{violation}作出什么处罚",
    "请根据历史案例推断合适的罚款金额",
    "这种情节轻微的{violation}会怎么处理",
    "如果情节恶劣，{violation}可能涉及刑事责任吗",
    "给一个完整的{violation}处罚建议包含罚款和警告",
)

TREND_TEMPLATES: tuple[str, ...] = (
    "近{year}年{violation}的处罚趋势如何",
    "统计不同年份{violation}的处罚方式变化",
    "{violation}的处罚分布近几年有什么变化",
    "最近{year_short}年{violation}案件数量怎么样",
    "按年度统计{violation}的处罚数量",
    "{agency}近年对{violation}的执法力度趋势",
    "从{year_past}到{year}年{violation}案件数量变化",
    "按年份画一下{violation}案件数量分布",
    "过去{year_short}年{violation}的处罚金额趋势如何",
    "{violation}与{violation_b}哪种违规更多",
    "近几年证券类违规整体走势怎么样",
    "帮我统计一下各年份{violation}的数量",
    "近{year_short}年{agency}处罚了多少家公司",
    "整体来看{violation}的处罚频率上升还是下降",
    "帮我汇总{year_past}-{year}年的{violation}情况",
    "按季度统计{violation}的处罚分布",
    "请画出{violation}年度趋势图",
    "近{year_short}年内幕交易和市场操纵数量对比",
    "{agency}各年度立案数量变化情况",
    "按违规类型画饼图看看分布",
    "不同地区证监局对{violation}的执法次数",
    "统计一下近年{violation}罚款金额中位数",
    "{violation}的处罚强度随时间有没有变化",
    "过去十年证券违规执法趋势如何",
    "{violation}在近{year_short}年占比变化",
)

TREND_VARS = {
    "year_short": ("三", "五", "十", "二十", "3", "5", "10", "15"),
    "year_past": tuple(str(y) for y in range(2000, 2022)),
    "violation_b": VIOLATIONS,
}


# --------------------------------------------------------------------------- #
# Expansion engine                                                             #
# --------------------------------------------------------------------------- #


def _fill(
    template: str,
    rng: random.Random,
    *,
    companies: list[str],
    extra: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """Substitute all ``{name}`` slots in ``template`` with a random pick."""
    if "{" not in template:
        return template
    vars_pool: dict[str, tuple[str, ...] | list[str]] = {
        "violation": VIOLATIONS,
        "year": YEARS,
        "law": LAWS,
        "agency": AGENCIES,
        "company": companies,
    }
    if extra:
        vars_pool.update(extra)

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        pool = vars_pool.get(key)
        if pool is None:
            return match.group(0)
        return rng.choice(list(pool))

    return re.sub(r"\{(\w+)\}", _replace, template)


def _expand_class(
    templates: tuple[str, ...],
    *,
    companies: list[str],
    extra: dict[str, tuple[str, ...]] | None,
    seed: int,
    rows_per_class: int = ROWS_PER_CLASS,
    variants_per_template: int = VARIANTS_PER_TEMPLATE,
) -> list[str]:
    """Expand ``templates`` to exactly ``rows_per_class`` unique-ish rows.

    For each template we generate ``variants_per_template`` filled strings;
    duplicates across templates are allowed (they act as natural label
    reinforcement), but we cap the output at ``rows_per_class`` rows.
    """
    if len(templates) * variants_per_template < rows_per_class:
        raise ValueError(
            f"templates({len(templates)}) * variants({variants_per_template}) "
            f"< rows_per_class({rows_per_class})"
        )

    rng = random.Random(seed)
    out: list[str] = []
    # For templates without slots, produce all variants by sampling sentence-level
    # jitters via the ``extra`` vocab.
    for template in templates[:TEMPLATES_PER_CLASS]:
        for _ in range(variants_per_template):
            text = _fill(template, rng, companies=companies, extra=extra)
            out.append(text)
    rng.shuffle(out)
    return out[:rows_per_class]


def build_training_set(companies: list[str], seed: int = RANDOM_SEED) -> list[dict[str, str]]:
    """Return a list of ``{"text", "label"}`` records, 500 per class * 7."""
    rows: list[dict[str, str]] = []

    class_configs: list[tuple[str, tuple[str, ...], dict[str, tuple[str, ...]] | None]] = [
        ("greeting", GREETING_TEMPLATES, GREETING_VARS),
        ("chitchat", CHITCHAT_TEMPLATES, CHITCHAT_VARS),
        ("out_of_scope", OUT_OF_SCOPE_TEMPLATES, OOS_VARS),
        ("case_retrieval", CASE_RETRIEVAL_TEMPLATES, None),
        ("law_grounding", LAW_GROUNDING_TEMPLATES, None),
        ("sanction_recommendation", SANCTION_TEMPLATES, None),
        ("trend_analysis", TREND_TEMPLATES, TREND_VARS),
    ]

    for idx, (label, templates, extra) in enumerate(class_configs):
        texts = _expand_class(
            templates,
            companies=companies,
            extra=extra,
            seed=seed + idx,
        )
        for text in texts:
            rows.append({"text": text, "label": label})

    return rows


def merge_seeds(
    rows: list[dict[str, str]],
    seeds_path: Path,
) -> list[dict[str, str]]:
    """Ensure every seed example is present (for label-leak sanity)."""
    if not seeds_path.exists():
        logger.warning("Seed file %s not found; skipping seed merge.", seeds_path)
        return rows
    payload = json.loads(seeds_path.read_text(encoding="utf-8"))
    existing = {(r["text"], r["label"]) for r in rows}
    for label, examples in payload.items():
        if label.startswith("_"):
            continue
        if not isinstance(examples, list):
            continue
        for example in examples:
            if not isinstance(example, str):
                continue
            key = (example.strip(), label)
            if key not in existing:
                rows.append({"text": key[0], "label": key[1]})
                existing.add(key)
    return rows


def write_jsonl(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    companies = load_top_companies(args.corpus, limit=100)
    logger.info("Loaded %d company candidates from %s", len(companies), args.corpus)

    rows = build_training_set(companies, seed=args.seed)
    rows = merge_seeds(rows, args.seeds)

    write_jsonl(rows, args.output)

    counter: Counter[str] = Counter(row["label"] for row in rows)
    logger.info("Wrote %d rows -> %s", len(rows), args.output)
    logger.info("Per-class counts: %s", dict(counter))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

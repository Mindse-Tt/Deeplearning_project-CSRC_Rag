"""意图注册表与路由层 —— 把 L1 分类结果落地为可执行的检索策略。

本模块衔接 L1 意图分类器（intent_model）与下游检索/聚合逻辑：

  * ``IntentSpec`` 描述每个意图对应的检索单元、top_k、回答区块等执行参数，
    全部由 ``configs/intents.json`` 配置驱动，便于不改代码即可调策略；
  * ``route_query`` 是统一入口：优先采用 ML 分类器的预测，
    当模型缺失或预测标签不在注册表内时，回退到关键词启发式路由（兜底），
    保证系统在无模型产物时仍可降级运行。

这是我们设计的"配置即策略"路由层：意图与检索行为解耦，新增意图只需改配置。
"""
from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

from csrc_rag.orchestration.intent_model import load_intent_classifier
from csrc_rag.settings import CONFIG_DIR


@dataclass(frozen=True)
class IntentSpec:
    name: str
    description: str
    retrieval_unit: str
    top_k: int
    response_sections: list[str]


@dataclass(frozen=True)
class IntentDecision:
    spec: IntentSpec
    confidence: float
    method: str
    scores: dict[str, float]


def load_registry(path: str | Path | None = None) -> dict[str, IntentSpec]:
    config_path = Path(path) if path else CONFIG_DIR / "intents.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        name: IntentSpec(
            name=name,
            description=config["description"],
            retrieval_unit=config["retrieval_unit"],
            top_k=config["top_k"],
            response_sections=config["response_sections"],
        )
        for name, config in payload.items()
    }


@lru_cache(maxsize=1)
def _load_router():
    # 分类器产物加载一次即缓存，避免每次请求重复反序列化 pickle。
    return load_intent_classifier()


def _heuristic_route(query: str, registry: dict[str, IntentSpec]) -> IntentDecision:
    # 关键词启发式兜底：当 ML 模型不可用时，按领域关键词命中顺序判定意图。
    # 优先级体现业务判断——处罚建议类 > 法条依据类 > 趋势统计类 > 默认案例检索。
    if any(token in query for token in ["处罚", "建议", "推荐", "罚款", "市场禁入"]):
        spec = registry["sanction_recommendation"]
        return IntentDecision(spec=spec, confidence=0.92, method="heuristic", scores={spec.name: 0.92})
    if any(token in query for token in ["法条", "法规", "依据", "违反"]):
        spec = registry["law_grounding"]
        return IntentDecision(spec=spec, confidence=0.9, method="heuristic", scores={spec.name: 0.9})
    if any(token in query for token in ["趋势", "统计", "分布", "近年", "变化"]):
        spec = registry["trend_analysis"]
        return IntentDecision(spec=spec, confidence=0.88, method="heuristic", scores={spec.name: 0.88})
    spec = registry["case_retrieval"]
    return IntentDecision(spec=spec, confidence=0.75, method="heuristic", scores={spec.name: 0.75})


def route_query(query: str, registry: dict[str, IntentSpec]) -> IntentDecision:
    # 路由主入口：ML 优先，启发式兜底。
    router = _load_router()
    if router is not None:
        prediction = router.predict(query)
        # 仅当预测标签存在于注册表中才采纳，否则退化到启发式（防止脏标签穿透）。
        if prediction.name in registry:
            return IntentDecision(
                spec=registry[prediction.name],
                confidence=prediction.confidence,
                method=prediction.method,
                scores=prediction.scores,
            )
    return _heuristic_route(query, registry)

"""
FastAPI v2 server — SIGNATURE-ONLY SKELETON.

This module wires the seven-layer pipeline (L0..L7) behind a FastAPI app
but contains NO implementation. Every layer is a `...` placeholder that
downstream agents will fill.

Run (once implemented):
    uvicorn csrc_rag.api.server_v2:app --host 127.0.0.1 --port 8000 --app-dir src

See docs/strategies/11-engineering-strategy.md for the full contract.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from csrc_rag.api.schemas import (
    ChartSpec,
    ErrorResponse,
    EventCard,
    FeedbackRequest,
    HealthResponse,
    IntentName,
    IntentsResponse,
    QueryPlan,
    QueryRequest,
    QueryResponse,
    RetrievalTrace,
    Slots,
    Timings,
    TrendBucket,
    TrendPayload,
    TrendRequest,
    ValidatorResult,
)

logger = logging.getLogger("csrc_rag.api")

# ---------------------------------------------------------------------------
# App-wide state
# ---------------------------------------------------------------------------


class AppState:
    """Singleton container for loaded models / indexes.

    Filled in ``lifespan``; every handler grabs it via ``Depends(get_state)``.
    """

    retrieval_engine: Any | None = None
    reranker: Any | None = None
    responder: Any | None = None
    query_rewriter: Any | None = None
    intent_router: Any | None = None
    validator: Any | None = None
    trend_analyzer: Any | None = None
    last_trace: QueryResponse | None = None  # /api/debug/last


STATE = AppState()


def get_state() -> AppState:
    return STATE


# ---------------------------------------------------------------------------
# Lifespan: load indexes / models once at boot.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load retrieval index / intent model / reranker / LoRA adapter."""
    ...
    yield
    # teardown if needed
    ...


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="CSRC RAG API v2",
        version="2.0.0",
        description="Seven-layer RAG pipeline for CSRC punishment cases.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CSRC_ALLOW_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static web/ at root (AFTER defining /api/* routes below).
    _register_routes(app)
    web_dir = Path(__file__).resolve().parents[3] / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


# ---------------------------------------------------------------------------
# Layer primitives — all STUBS.
# ---------------------------------------------------------------------------


async def l0_preprocess(query: str) -> tuple[str, list[str]]:
    """L0: normalize + truncate. Returns (clean_query, warnings)."""
    ...


async def l1_intent(clean_query: str, forced: IntentName | None) -> tuple[IntentName, float, str, dict[str, float]]:
    """L1: route to one of 7 intents. Returns (intent, confidence, method, scores)."""
    ...


async def l2_rewrite(
    clean_query: str,
    history: list[dict[str, str]],
    intent: IntentName,
) -> tuple[str, Slots, list[str]]:
    """L2: coref + synonym + slot extraction. Returns (rewritten, slots, synonyms)."""
    ...


async def l3_retrieve(
    rewritten_query: str,
    plan: QueryPlan,
    state: AppState,
) -> tuple[list[EventCard], RetrievalTrace]:
    """L3: BM25 ∥ Dense → RRF → Reranker. Returns (ranked_events, full_trace)."""
    ...


async def l4_pack_evidence(
    events: list[EventCard],
    max_tokens: int,
) -> list[EventCard]:
    """L4: truncate / summarise evidence so it fits in the generator context."""
    ...


async def l5_generate(
    clean_query: str,
    intent: IntentName,
    evidence: list[EventCard],
    history: list[dict[str, str]],
    state: AppState,
) -> tuple[str, list[str], list[str], str, str | None]:
    """L5: LLM generation.
    Returns (answer, cited_event_ids, cited_laws, backend, model_name).
    """
    ...


async def l6_trend(
    slots: Slots,
    events: list[EventCard],
    state: AppState,
) -> TrendPayload:
    """L6: aggregate events by year / type → buckets + ChartSpec."""
    ...


async def l7_validate(
    answer: str,
    cited_event_ids: list[str],
    cited_laws: list[str],
    evidence: list[EventCard],
    state: AppState,
) -> ValidatorResult:
    """L7: check every claim against evidence whitelist."""
    ...


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    async def health(state: AppState = Depends(get_state)) -> HealthResponse:
        ...

    @app.get("/api/intents", response_model=IntentsResponse, tags=["meta"])
    async def intents(state: AppState = Depends(get_state)) -> IntentsResponse:
        ...

    @app.post(
        "/api/query",
        response_model=QueryResponse,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
        tags=["rag"],
    )
    async def query(body: QueryRequest, state: AppState = Depends(get_state)) -> QueryResponse:
        """Main entry. Runs L0..L7 sequentially with per-layer degrade guards."""
        request_id = f"req-{uuid.uuid4().hex[:10]}"
        t0 = time.perf_counter()
        warnings: list[str] = []
        response = QueryResponse(
            request_id=request_id,
            session_id=body.session_id,
            query_original=body.query,
        )
        # Each layer below wraps its call in try/except and writes into `response`.
        # L0:
        ...
        # L1:
        ...
        # L2:
        ...
        # L3 (build plan → retrieve):
        ...
        # L4:
        ...
        # L5:
        ...
        # L6 (only when intent == trend_analysis):
        ...
        # L7:
        ...
        response.warnings = warnings
        response.timings_ms.total = int((time.perf_counter() - t0) * 1000)
        STATE.last_trace = response
        return response

    @app.post("/api/query/stream", tags=["rag"])
    async def query_stream(body: QueryRequest, state: AppState = Depends(get_state)) -> StreamingResponse:
        """SSE variant. Streams L5 tokens, then sends a final `done` event
        carrying the same QueryResponse JSON as /api/query."""

        async def event_gen() -> AsyncIterator[bytes]:
            ...
            yield b""

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.get("/api/trend", response_model=TrendPayload, tags=["rag"])
    async def trend(
        year_from: int = 2019,
        year_to: int = 2025,
        violation: str | None = None,
        state: AppState = Depends(get_state),
    ) -> TrendPayload:
        """Standalone L6 endpoint for the trend tab (no LLM)."""
        ...

    @app.post("/api/feedback", tags=["meta"])
    async def feedback(body: FeedbackRequest) -> dict[str, bool]:
        ...

    @app.get("/api/debug/last", response_model=QueryResponse, tags=["meta"])
    async def debug_last(state: AppState = Depends(get_state)) -> QueryResponse:
        if os.environ.get("CSRC_DEBUG", "0") != "1":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="debug disabled")
        if state.last_trace is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no trace")
        return state.last_trace

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(exc)).model_dump(),
        )


# ---------------------------------------------------------------------------
# Module-level app instance (for `uvicorn csrc_rag.api.server_v2:app`).
# ---------------------------------------------------------------------------

app = create_app()


__all__ = [
    "AppState",
    "app",
    "create_app",
    "get_state",
    "l0_preprocess",
    "l1_intent",
    "l2_rewrite",
    "l3_retrieve",
    "l4_pack_evidence",
    "l5_generate",
    "l6_trend",
    "l7_validate",
]

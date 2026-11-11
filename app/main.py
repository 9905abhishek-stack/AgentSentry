"""FastAPI routes for executing and observing audits."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client.http.exceptions import ApiException

from app import __version__
from app.config import settings
from app.graph.nodes.retriever import close_corpus
from app.graph.workflow import audit_dag
from app.schemas.agent_state import AgentState, AuditResponse, QueryRequest
from app.services.cache import cache_service
from app.services.corpus import compatible_collection, connect_corpus
from app.services.vllm_client import vllm_service

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AuditMetrics:
    total_audits: int = 0
    cache_hits: int = 0
    hitl_escalations: int = 0
    total_latency_ms: float = 0.0

    def record(self, response: AuditResponse) -> None:
        self.total_audits += 1
        self.cache_hits += int(response.cache_hit)
        self.hitl_escalations += int(response.verdict.requires_hitl)
        self.total_latency_ms += response.latency_ms


metrics = AuditMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cache_service.init()
    try:
        yield
    finally:
        await asyncio.gather(
            cache_service.close(), vllm_service.close(), asyncio.to_thread(close_corpus)
        )


app = FastAPI(
    title="AgentSentry API",
    description="Evidence-grounded claim auditing",
    version=__version__,
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


def corpus_health() -> str:
    if not settings.qdrant_host:
        return "disabled"
    try:
        client = connect_corpus()
        try:
            collection = settings.qdrant_collection
            if not client.collection_exists(collection):
                return "missing_collection"
            params = client.get_collection(collection).config.params
            return (
                "available"
                if compatible_collection(params)
                else "incompatible_collection"
            )
        finally:
            client.close()
    except (ApiException, OSError, ValueError, RuntimeError):
        return "unavailable"


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    corpus, redis, inference = await asyncio.gather(
        asyncio.to_thread(corpus_health), cache_service.health(), vllm_service.health()
    )
    subsystems = {"qdrant": corpus, "redis": redis, "vllm": inference}
    return {
        "status": "healthy"
        if set(subsystems.values()) <= {"available", "disabled"}
        else "degraded",
        "app": settings.app_name,
        "environment": settings.app_env,
        "evidence_locked_strict": settings.evidence_locked_strict,
        "subsystems": subsystems,
    }


@app.get("/api/v1/metrics", tags=["Observability"])
async def get_metrics() -> dict[str, int | float]:
    denominator = max(metrics.total_audits, 1)
    return {
        "total_audits": metrics.total_audits,
        "cache_hits": metrics.cache_hits,
        "cache_hit_rate": round(metrics.cache_hits / denominator, 3),
        "hitl_escalation_count": metrics.hitl_escalations,
        "average_latency_ms": round(metrics.total_latency_ms / denominator, 2),
    }


async def run_audit(
    request: QueryRequest,
) -> AsyncGenerator[dict[str, str] | AuditResponse, None]:
    started = time.perf_counter()
    strict = (
        settings.evidence_locked_strict
        if request.strict_evidence_lock is None
        else request.strict_evidence_lock
    )
    key = None
    response = None
    if (
        request.context_documents is None
        and settings.qdrant_host
        and vllm_service.base_url
    ):
        key = cache_service.key(request.query, strict)
        response = await cache_service.get(key)

    if response is None:
        initial = AgentState(
            query=request.query,
            strict_evidence_lock=strict,
            context_documents=request.context_documents,
        )
        final = None
        async for mode, output in audit_dag.astream(
            initial, stream_mode=["updates", "values"]
        ):
            if mode == "values":
                final = output
            elif mode == "updates" and isinstance(output, dict):
                for node, update in output.items():
                    yield {
                        "event": "node_complete",
                        "node": node,
                        "latest_log": update["execution_log"][-1],
                    }

        state = AgentState.model_validate(final)
        verdict = state.audit_verdict
        if verdict is None:
            raise RuntimeError("Audit workflow completed without a verdict.")
        evidence = {
            item.id: item
            for items in state.retrieved_evidence.values()
            for item in items
        }
        response = AuditResponse(
            query=request.query,
            status=verdict.overall_status,
            verdict=verdict,
            evidence_items=list(evidence.values()),
            latency_ms=0.0,
        )
    else:
        response.cache_hit = True

    response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
    response.verdict.total_latency_ms = response.latency_ms
    if (
        key is not None
        and not response.cache_hit
        and not response.verdict.requires_hitl
    ):
        await cache_service.set(key, response)
    metrics.record(response)
    yield response


@app.post("/api/v1/audit", response_model=AuditResponse, tags=["Audit"])
async def execute_audit(request: QueryRequest) -> AuditResponse:
    try:
        async with aclosing(run_audit(request)) as audit:
            async for item in audit:
                if isinstance(item, AuditResponse):
                    return item
    except Exception as exc:
        logger.exception("Audit workflow failed.")
        raise HTTPException(status_code=500, detail="Audit workflow failed.") from exc
    raise HTTPException(
        status_code=500, detail="Audit workflow did not return a result."
    )


@app.post("/api/v1/audit/stream", tags=["Audit"])
async def stream_audit(request: QueryRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        yield sse({"event": "start", "query": request.query})
        try:
            async with aclosing(run_audit(request)) as audit:
                async for item in audit:
                    if isinstance(item, AuditResponse):
                        yield sse(
                            {"event": "done", "result": item.model_dump(mode="json")}
                        )
                    else:
                        yield sse(item)
        except Exception:
            logger.exception("Streaming audit workflow failed.")
            yield sse({"event": "error", "message": "Audit workflow failed."})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, reload=settings.debug
    )

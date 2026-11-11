"""
AgentSentry FastAPI Production Service
Exposes REST and SSE Streaming endpoints for real-time evidence-locked AI audits.
"""
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas.agent_state import (
    QueryRequest,
    AuditResponse,
    AgentState,
    VerdictEnum
)
from app.services.cache import cache_service
from app.services.vllm_client import vllm_service
from app.graph.workflow import audit_dag

# Setup logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("agentsentry.api")

# In-memory operational metrics
METRICS = {
    "total_audits": 0,
    "cache_hits": 0,
    "escalated_hitl": 0,
    "total_latency_ms": 0.0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing AgentSentry audit engine...")
    await cache_service.init()
    yield
    logger.info("Shutting down AgentSentry services...")
    await cache_service.close()
    await vllm_service.close()


app = FastAPI(
    title="AgentSentry API",
    description="Enterprise Evidence-Locked Multi-Agent Audit System",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    """Returns the operational status of AgentSentry subsystems."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "environment": settings.app_env,
        "vllm_endpoint": settings.vllm_api_base,
        "qdrant_host": settings.qdrant_host,
        "evidence_locked_strict": settings.evidence_locked_strict,
    }


@app.get("/api/v1/metrics", tags=["Observability"])
async def get_metrics():
    """Returns real-time SLA and latency metrics."""
    total = max(METRICS["total_audits"], 1)
    return {
        "total_audits": METRICS["total_audits"],
        "cache_hits": METRICS["cache_hits"],
        "cache_hit_rate": round(METRICS["cache_hits"] / total, 3),
        "hitl_escalation_count": METRICS["escalated_hitl"],
        "average_latency_ms": round(METRICS["total_latency_ms"] / total, 2),
    }


@app.post("/api/v1/audit", response_model=AuditResponse, tags=["Audit"])
async def execute_audit(request: QueryRequest):
    """
    Executes an evidence-locked audit through the LangGraph StateGraph DAG.
    1. Checks Redis semantic cache.
    2. Decomposes query into atomic claims.
    3. Executes dense + sparse hybrid retrieval with RRF.
    4. Evaluates evidence against the Frozen Intermediate Representation (FIR).
    5. Computes confidence and escalates to HITL if below SLA.
    """
    start_time = time.perf_counter()
    METRICS["total_audits"] += 1

    # 1. Cache Check
    cached_verdict = await cache_service.get(request.query)
    if cached_verdict:
        METRICS["cache_hits"] += 1
        elapsed = (time.perf_counter() - start_time) * 1000.0
        cached_verdict.total_latency_ms = round(elapsed, 2)
        return AuditResponse(
            query=request.query,
            status=cached_verdict.overall_status,
            verdict=cached_verdict,
            evidence_items=[],
            latency_ms=round(elapsed, 2),
            cache_hit=True
        )

    # 2. Initialize StateGraph
    initial_state = AgentState(query=request.query)

    try:
        # Run asynchronous DAG
        raw_state = await audit_dag.ainvoke(initial_state)
        final_state: AgentState = AgentState(**raw_state) if isinstance(raw_state, dict) else raw_state
    except Exception as e:
        logger.error(f"Error during audit DAG execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audit graph execution failed: {str(e)}"
        )

    elapsed = (time.perf_counter() - start_time) * 1000.0
    METRICS["total_latency_ms"] += elapsed

    verdict = final_state.audit_verdict
    if verdict:
        verdict.total_latency_ms = round(elapsed, 2)
        if verdict.requires_hitl:
            METRICS["escalated_hitl"] += 1

        # Cache high-confidence verdicts
        if not verdict.requires_hitl:
            await cache_service.set(request.query, verdict)

    # Collect all flattened evidence citations
    all_evidence = []
    for claim_ev in final_state.retrieved_evidence.values():
        all_evidence.extend(claim_ev)

    return AuditResponse(
        query=request.query,
        status=verdict.overall_status if verdict else VerdictEnum.INSUFFICIENT_EVIDENCE,
        verdict=verdict,
        evidence_items=all_evidence,
        latency_ms=round(elapsed, 2),
        cache_hit=False
    )


@app.post("/api/v1/audit/stream", tags=["Audit"])
async def stream_audit(request: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint:
    Streams step-by-step audit progress as each node in the LangGraph DAG completes.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = AgentState(query=request.query)
        yield f"data: {json.dumps({'event': 'start', 'query': request.query})}\n\n"

        async for output in audit_dag.astream(initial_state):
            for node_name, state_update in output.items():
                payload = {
                    "event": "node_complete",
                    "node": node_name,
                    "latest_log": state_update.execution_log[-1] if state_update.execution_log else ""
                }
                yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

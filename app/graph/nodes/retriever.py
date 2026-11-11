"""Retrieve evidence from request documents or the configured Qdrant corpus."""

import asyncio
import hashlib
import logging
from threading import Lock
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException

from app.config import settings
from app.schemas.agent_state import (
    MAX_CONTEXT_CHUNKS,
    AgentState,
    EvidenceChunk,
    EvidenceItem,
)
from app.services.corpus import (
    DENSE_MODEL,
    DENSE_VECTOR,
    SPARSE_MODEL,
    SPARSE_VECTOR,
    connect_corpus,
)

logger = logging.getLogger(__name__)
_corpus_client: QdrantClient | None = None
_corpus_lock = Lock()


def close_corpus() -> None:
    global _corpus_client
    with _corpus_lock:
        if _corpus_client is not None:
            _corpus_client.close()
            _corpus_client = None


def compute_rrf(
    dense_items: list[EvidenceItem],
    sparse_items: list[EvidenceItem],
    k: int = 60,
) -> list[EvidenceItem]:
    scores: dict[str, float] = {}
    items: dict[str, EvidenceItem] = {}
    for ranked, score_field in (
        (dense_items, "dense_score"),
        (sparse_items, "sparse_score"),
    ):
        seen = set()
        for rank, item in enumerate(ranked, 1):
            if item.id in seen:
                continue
            seen.add(item.id)
            scores[item.id] = scores.get(item.id, 0.0) + 1 / (k + rank)
            previous = items.get(item.id, item)
            items[item.id] = previous.model_copy(
                update={score_field: getattr(item, score_field)}
            )
    return [
        items[point_id].model_copy(update={"fused_rrf_score": scores[point_id]})
        for point_id in sorted(
            scores, key=lambda point_id: (-scores[point_id], point_id)
        )
    ]


def search_corpus(text: str) -> list[EvidenceItem]:
    global _corpus_client
    requests = [
        models.QueryRequest(
            query=models.Document(text=text, model=model),
            using=vector,
            limit=MAX_CONTEXT_CHUNKS,
            with_payload=True,
        )
        for vector, model in (
            (DENSE_VECTOR, DENSE_MODEL),
            (SPARSE_VECTOR, SPARSE_MODEL),
        )
    ]
    # The client lazily loads its embedding models on first use.
    with _corpus_lock:
        if _corpus_client is None:
            _corpus_client = connect_corpus()
        dense, sparse = _corpus_client.query_batch_points(
            settings.qdrant_collection, requests
        )

    def to_evidence(
        points: list[models.ScoredPoint], score_field: str
    ) -> list[EvidenceItem]:
        results = []
        for point in points:
            chunk = EvidenceChunk.model_validate(point.payload)
            revision = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            payload = {
                **chunk.model_dump(),
                "id": f"{point.id}:{revision}",
                score_field: point.score,
            }
            results.append(EvidenceItem.model_validate(payload))
        return results

    return compute_rrf(
        to_evidence(dense.points, "dense_score"),
        to_evidence(sparse.points, "sparse_score"),
        k=settings.hybrid_rrf_k,
    )[:MAX_CONTEXT_CHUNKS]


async def retrieve_evidence(state: AgentState) -> dict[str, Any]:
    claim = state.claims[state.current_claim_index]
    review = state.requires_human_review
    failures = state.failed_retrievals
    evidence = []
    if state.context_documents is not None:
        evidence = [
            EvidenceItem(
                id=f"context-{index}", source_doc="request", chunk_id=index, text=text
            )
            for index, text in enumerate(state.context_documents, 1)
        ]
        message = f"Loaded {len(evidence)} request evidence chunk(s)."
    elif settings.qdrant_host:
        try:
            evidence = await asyncio.to_thread(search_corpus, claim.text)
            message = f"Retrieved {len(evidence)} evidence chunk(s) from the corpus."
        except (ApiException, OSError, ValueError, RuntimeError) as exc:
            logger.warning("Corpus retrieval failed (%s)", type(exc).__name__)
            message = "Corpus retrieval failed. Human review required."
            review = True
            failures = [*failures, claim.claim_id]
    else:
        message = "No evidence corpus configured."

    return {
        "retrieved_evidence": {**state.retrieved_evidence, claim.claim_id: evidence},
        "requires_human_review": review,
        "failed_retrievals": failures,
        "execution_log": [*state.execution_log, message],
    }

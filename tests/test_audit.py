import asyncio
import hashlib
import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import QueryResponse
from redis.exceptions import ConnectionError as RedisConnectionError

from app import main
from app.config import settings
from app.graph.nodes import retriever, verifier
from app.graph.nodes.planner import order_claims
from app.schemas.agent_state import (
    AuditResponse,
    Claim,
    EvidenceChunk,
    EvidenceItem,
    VerdictEnum,
    VerificationIssue,
    VerificationResult,
)
from app.services import cache as cache_module
from app.services.cache import CacheService
from app.services.corpus import DENSE_SIZE, DENSE_VECTOR, SPARSE_VECTOR, index_chunks
from app.services.vllm_client import InferenceUnavailable, VLLMClient, vllm_service
from evaluation import run_ragas_eval

QUERY = "Production logs are retained for 30 days."
EVIDENCE = "Production logs are retained for 30 days under the retention policy."


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "qdrant_host", None)
    monkeypatch.setattr(settings, "redis_url", None)
    monkeypatch.setattr(settings, "evidence_locked_strict", True)
    monkeypatch.setattr(settings, "max_verification_retries", 1)
    monkeypatch.setattr(settings, "confidence_threshold", 0.85)
    monkeypatch.setattr(vllm_service, "base_url", None)
    monkeypatch.setattr(main, "metrics", main.AuditMetrics())
    with TestClient(main.app) as session:
        yield session


def fact(evidence_id: str = "context-1", quote: str = EVIDENCE) -> dict[str, str]:
    return {
        "entity": "production logs",
        "relation": "retention",
        "value": "30 days",
        "evidence_id": evidence_id,
        "quote": quote,
    }


def install_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extraction: list[dict[str, str]] | None = None,
    citation: str | list[str] = "context-1",
    decision: str = "SUPPORTED",
    confidence: float = 0.97,
    plan: list[dict[str, Any]] | None = None,
) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    async def generate(messages: list[dict[str, str]]) -> str:
        prompt, user = (item["content"] for item in messages)
        if prompt.startswith("Decompose"):
            calls.append(("plan", user))
            return json.dumps(
                {
                    "claims": plan
                    if plan is not None
                    else [{"claim_id": "c1", "text": user}]
                }
            )
        payload = json.loads(user)
        if prompt.startswith(verifier.EXTRACTION_PROMPT):
            calls.append(("extract", payload))
            return json.dumps(
                {"facts": extraction if extraction is not None else [fact()]}
            )
        assert prompt.startswith(verifier.DECISION_PROMPT)
        calls.append(("decide", payload))
        return json.dumps(
            {
                "verdict": decision,
                "confidence": confidence,
                "citation_ids": [citation] if isinstance(citation, str) else citation,
                "rationale": "The decision follows the supplied facts.",
            }
        )

    monkeypatch.setattr(vllm_service, "generate", generate)
    return calls


def test_offline_audit_does_not_invent_evidence(client: TestClient) -> None:
    response = client.post("/api/v1/audit", json={"query": QUERY})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["verdict"]["requires_hitl"]
    assert result["evidence_items"] == []
    assert result["verdict"]["verified_claims"][0]["citation_ids"] == []
    assert not result["cache_hit"]
    assert client.post("/api/v1/audit", json={"query": "     "}).status_code == 422
    assert client.get("/health").json()["subsystems"] == {
        "qdrant": "disabled",
        "redis": "disabled",
        "vllm": "disabled",
    }

    with_evidence = client.post(
        "/api/v1/audit", json={"query": QUERY, "context_documents": [EVIDENCE]}
    ).json()
    assert with_evidence["status"] == "INSUFFICIENT_EVIDENCE"
    assert with_evidence["verdict"]["requires_hitl"]
    assert with_evidence["evidence_items"][0]["text"] == EVIDENCE


def test_frozen_facts_have_order_independent_hashes() -> None:
    items = [
        EvidenceItem(id="e1", source_doc="one.txt", text=EVIDENCE),
        EvidenceItem(id="e2", source_doc="two.txt", text="Access logs are retained."),
    ]
    facts = [fact("e1"), fact("e2", items[1].text)]
    baseline = verifier.freeze_facts(json.dumps({"facts": facts}), items)
    reversed_result = verifier.freeze_facts(
        json.dumps({"facts": [facts[1], facts[0], facts[0]]}), items[::-1]
    )
    assert baseline == reversed_result
    changed_items = [
        items[0].model_copy(update={"text": EVIDENCE + " Updated."}),
        items[1],
    ]
    assert (
        verifier.freeze_facts(json.dumps({"facts": facts}), changed_items)[1]
        != baseline[1]
    )


@pytest.mark.parametrize("strict", [True, False])
def test_freeze_then_decide(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    calls = install_model(monkeypatch)
    result = client.post(
        "/api/v1/audit",
        json={
            "query": QUERY,
            "context_documents": [EVIDENCE],
            "strict_evidence_lock": strict,
        },
    ).json()
    assert result["status"] == "SUPPORTED"
    assert not result["verdict"]["requires_hitl"]
    assert [stage for stage, _ in calls] == ["plan", "extract", "decide"]
    assert ("evidence" not in calls[-1][1]) is strict
    claim = result["verdict"]["verified_claims"][0]
    assert claim["citation_ids"] == [result["evidence_items"][0]["id"]]
    assert (
        claim["frozen_ir"][0]["evidence_sha256"]
        == hashlib.sha256(EVIDENCE.encode()).hexdigest()
    )
    canonical = json.dumps(
        claim["frozen_ir"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    assert claim["frozen_ir_hash"] == hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize(
    ("extraction", "citation", "decision", "failed_stage"),
    [
        ([fact(quote="Invented quote.")], "context-1", "SUPPORTED", "extract"),
        ([fact(evidence_id="other")], "context-1", "SUPPORTED", "extract"),
        ([fact()], "missing", "SUPPORTED", "decide"),
        ([fact()], "context-1", "CONFLICTING_EVIDENCE", "decide"),
        ([fact()], "context-1", "UNRECOGNIZED", "decide"),
    ],
)
def test_invalid_model_evidence_and_verdicts_abstain(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    extraction: list[dict[str, str]],
    citation: str,
    decision: str,
    failed_stage: str,
) -> None:
    calls = install_model(
        monkeypatch, extraction=extraction, citation=citation, decision=decision
    )
    result = client.post(
        "/api/v1/audit", json={"query": QUERY, "context_documents": [EVIDENCE]}
    ).json()
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["verdict"]["requires_hitl"]
    assert result["verdict"]["verified_claims"][0]["citation_ids"] == []
    assert [stage for stage, _ in calls].count(failed_stage) == 2


def test_graph_respects_dependencies_and_flags_uncertainty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install_model(
        monkeypatch,
        plan=[
            {
                "claim_id": "second",
                "text": "A dependent statement.",
                "dependencies": ["first"],
            },
            {"claim_id": "first", "text": QUERY},
        ],
        decision="REFUTED",
    )
    result = client.post(
        "/api/v1/audit", json={"query": QUERY, "context_documents": [EVIDENCE]}
    ).json()
    assert result["status"] == "REFUTED"
    assert result["verdict"]["requires_hitl"]
    results = result["verdict"]["verified_claims"]
    assert [item["claim_id"] for item in results] == ["first", "second"]
    assert [item["verdict"] for item in results] == ["REFUTED", "INSUFFICIENT_EVIDENCE"]
    assert [stage for stage, _ in calls] == ["plan", "extract", "decide", "decide"]


def test_duplicate_passages_do_not_establish_conflicting_evidence(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_model(
        monkeypatch,
        extraction=[fact("context-1"), fact("context-2")],
        citation=["context-1", "context-2"],
        decision="CONFLICTING_EVIDENCE",
    )
    result = client.post(
        "/api/v1/audit",
        json={"query": QUERY, "context_documents": [EVIDENCE, EVIDENCE]},
    ).json()
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["verdict"]["verified_claims"][0]["issue"] == "INVALID_MODEL_OUTPUT"


@pytest.mark.parametrize(
    ("operator", "expected"),
    [("or", "SUPPORTED"), ("and", "INSUFFICIENT_EVIDENCE")],
)
def test_overall_verdict_uses_original_assertion(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, operator: str, expected: str
) -> None:
    first = QUERY
    second = "Production logs are retained for 90 days."
    original = (
        "Production logs are retained for 30 days or 90 days."
        if operator == "or"
        else "Production logs are retained for 30 days and encrypted at rest."
    )

    async def generate(messages: list[dict[str, str]]) -> str:
        prompt, user = (item["content"] for item in messages)
        if prompt.startswith("Decompose"):
            planned = [{"claim_id": "first", "text": first}]
            if operator == "or":
                planned.append({"claim_id": "second", "text": second})
            return json.dumps({"claims": planned})
        payload = json.loads(user)
        if prompt.startswith(verifier.EXTRACTION_PROMPT):
            return json.dumps({"facts": [fact()]})
        assert prompt.startswith(verifier.DECISION_PROMPT)
        outcome = {first: "SUPPORTED", second: "REFUTED", original: expected}
        return json.dumps(
            {
                "verdict": outcome[payload["claim"]],
                "confidence": 0.97,
                "citation_ids": ["context-1"],
                "rationale": "Decided against the quoted retention policy.",
            }
        )

    monkeypatch.setattr(vllm_service, "generate", generate)
    result = client.post(
        "/api/v1/audit", json={"query": original, "context_documents": [EVIDENCE]}
    ).json()
    assert result["status"] == expected
    assertion = result["verdict"]["assertion_verification"]
    assert assertion["claim_text"] == original
    assert assertion["frozen_ir_hash"]
    assert result["verdict"]["requires_hitl"] is (operator == "and")


def test_maximum_claim_plan_completes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_model(
        monkeypatch, plan=[{"claim_id": f"c{i}", "text": QUERY} for i in range(10)]
    )
    result = client.post(
        "/api/v1/audit", json={"query": QUERY, "context_documents": [EVIDENCE]}
    ).json()
    assert result["status"] == "SUPPORTED"
    assert len(result["verdict"]["verified_claims"]) == 10


@pytest.mark.parametrize("dependencies", [["missing"], ["second"]])
def test_invalid_dependencies(dependencies: list[str]) -> None:
    claims = [
        Claim(claim_id="first", text=QUERY, dependencies=dependencies),
        Claim(claim_id="second", text=QUERY, dependencies=["first"]),
    ]
    with pytest.raises(ValueError):
        order_claims(claims)


def test_stream_cache_and_metrics(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = EvidenceItem(id="corpus-1", source_doc="policy.txt", text=EVIDENCE)
    monkeypatch.setattr(settings, "qdrant_host", "unused.test")
    monkeypatch.setattr(vllm_service, "base_url", "http://unused.test/v1")
    monkeypatch.setattr(retriever, "search_corpus", lambda _query: [evidence])
    calls = install_model(
        monkeypatch, extraction=[fact("corpus-1")], citation="corpus-1"
    )

    request = {"query": QUERY}
    result = client.post("/api/v1/audit", json=request).json()
    streamed = client.post("/api/v1/audit/stream", json=request)
    events = [
        json.loads(line[6:])
        for line in streamed.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["event"] for event in events] == ["start", "done"]
    assert events[-1]["result"]["cache_hit"]
    assert events[-1]["result"]["evidence_items"] == result["evidence_items"]
    assert [stage for stage, _ in calls] == ["plan", "extract", "decide"]

    assert not client.post(
        "/api/v1/audit", json={**request, "strict_evidence_lock": False}
    ).json()["cache_hit"]
    assert not client.post("/api/v1/audit", json={"query": QUERY.lower()}).json()[
        "cache_hit"
    ]
    explicit_empty = client.post(
        "/api/v1/audit", json={**request, "context_documents": []}
    ).json()
    assert (
        explicit_empty["status"] == "INSUFFICIENT_EVIDENCE"
        and not explicit_empty["cache_hit"]
    )
    counts = client.get("/api/v1/metrics").json()
    assert counts["total_audits"] == 5 and counts["cache_hits"] == 1
    assert counts["hitl_escalation_count"] == 1


def test_stream_miss_reports_final_verdict(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_model(monkeypatch)
    response = client.post(
        "/api/v1/audit/stream", json={"query": QUERY, "context_documents": [EVIDENCE]}
    )
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["node"] for event in events if event["event"] == "node_complete"] == [
        "planner",
        "retriever",
        "verifier",
        "supervisor",
    ]
    assert events[-1]["result"]["status"] == "SUPPORTED"


def test_local_cache_expiry_capacity_and_copy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = AuditResponse.model_validate(
        client.post("/api/v1/audit", json={"query": QUERY}).json()
    )
    monkeypatch.setattr(settings, "cache_ttl_seconds", 10)
    monkeypatch.setattr(settings, "cache_max_entries", 1)
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        cache_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    cache = CacheService()

    async def exercise() -> None:
        await cache.set("a", response)
        first = await cache.get("a")
        assert first is not None
        first.verdict.total_latency_ms = -1
        fresh = await cache.get("a")
        assert fresh is not None and fresh.verdict.total_latency_ms >= 0
        await cache.set("b", response)
        assert await cache.get("a") is None
        assert await cache.get("b") is not None
        clock.now += 10
        assert await cache.get("b") is None

    asyncio.run(exercise())


def test_redis_miss_can_use_fallback_after_recovery(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = AuditResponse.model_validate(
        client.post("/api/v1/audit", json={"query": QUERY}).json()
    )
    cache = CacheService()

    async def disconnected(*_args: Any, **_kwargs: Any) -> None:
        raise RedisConnectionError("unavailable")

    async def redis_get(_key: str) -> str | None:
        return None

    monkeypatch.setattr(
        cache, "_redis", SimpleNamespace(get=redis_get, set=disconnected)
    )

    async def exercise() -> None:
        await cache.set("fallback", response)
        assert await cache.get("fallback") == response

    asyncio.run(exercise())


def test_indexer_validates_later_rows_before_writing() -> None:
    with closing(QdrantClient(":memory:")) as client:

        def invalid_rows() -> Iterator[EvidenceChunk]:
            for index in range(32):
                yield EvidenceChunk(
                    source_doc="policy.txt", chunk_id=index, text=EVIDENCE
                )
            raise ValueError("Invalid evidence chunk on line 33.")

        with pytest.raises(ValueError, match="line 33"):
            index_chunks(client, "test", invalid_rows())
        assert not client.collection_exists("test")

        chunk = EvidenceChunk(source_doc="policy.txt", chunk_id=1, text=EVIDENCE)
        with pytest.raises(ValueError, match="Duplicate source and chunk ID"):
            index_chunks(client, "test", [chunk, chunk])
        assert not client.collection_exists("test")


def test_indexer_rejects_unconfirmed_qdrant_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection(
            "test",
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=DENSE_SIZE, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        monkeypatch.setattr(
            client,
            "upsert",
            lambda **_: models.UpdateResult(status=models.UpdateStatus.WAIT_TIMEOUT),
        )
        chunk = EvidenceChunk(source_doc="policy.txt", chunk_id=1, text=EVIDENCE)
        with pytest.raises(RuntimeError, match="after 0 confirmed chunk"):
            index_chunks(client, "test", [chunk])


def test_health_reports_incompatible_corpus(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        settings.qdrant_collection,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    monkeypatch.setattr(settings, "qdrant_host", "configured.test")
    monkeypatch.setattr(main, "connect_corpus", lambda: qdrant)
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["subsystems"]["qdrant"] == "incompatible_collection"


def test_evaluation_excludes_failed_inference_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def health() -> str:
        return "available"

    attempts = iter(
        [
            (
                VerdictEnum.INSUFFICIENT_EVIDENCE,
                VerificationIssue.INFERENCE_UNAVAILABLE,
            ),
            (VerdictEnum.INSUFFICIENT_EVIDENCE, None),
            (VerdictEnum.REFUTED, None),
            (VerdictEnum.SUPPORTED, None),
        ]
    )

    async def verify(
        claim: Claim, _evidence: list[EvidenceItem], strict: bool
    ) -> VerificationResult:
        assert strict
        verdict, issue = next(attempts)
        return VerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            verdict=verdict,
            confidence=0.9,
            rationale="fixture",
            issue=issue,
        )

    monkeypatch.setattr(vllm_service, "health", health)
    monkeypatch.setattr(run_ragas_eval, "verify_claim", verify)
    fixtures = run_ragas_eval.FIXTURES
    monkeypatch.setattr(run_ragas_eval, "FIXTURES", (fixtures[2], fixtures[1]))
    results = asyncio.run(run_ragas_eval.run_evaluation())
    assert results.failed_attempts == 1
    assert results.evaluated_baselines == results.evaluated_order_pairs == 1
    assert results.verdict_accuracy == results.evidence_order_flip_rate == 1


def test_vllm_preserves_api_base_and_checks_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "served"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    service = VLLMClient(base_url="http://vllm.test/prefix/v1", model="served")

    async def exercise() -> None:
        async with httpx.AsyncClient(
            base_url="http://vllm.test/prefix/v1/",
            transport=httpx.MockTransport(handle),
        ) as http:
            monkeypatch.setattr(service, "_client", http)
            assert await service.health() == "available"
            assert await service.generate([{"role": "user", "content": QUERY}]) == "{}"

    asyncio.run(exercise())
    assert paths == ["/prefix/v1/models", "/prefix/v1/chat/completions"]


def test_vllm_rejects_content_filtered_completions() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "content_filter", "message": {"content": "{}"}}
                ]
            },
        )

    async def exercise() -> None:
        service = VLLMClient(base_url="http://vllm.test/v1", model="served")
        async with httpx.AsyncClient(
            base_url="http://vllm.test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            service._client = client
            with pytest.raises(InferenceUnavailable):
                await service.generate([{"role": "user", "content": QUERY}])

    asyncio.run(exercise())


def test_corpus_pool_concurrent_searches_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.closed = False

        def query_batch_points(
            self, _collection: str, _requests: list[models.QueryRequest]
        ) -> tuple[QueryResponse, QueryResponse]:
            self.entered.set()
            assert self.release.wait(timeout=10.0), "Timed out waiting for release"
            response = QueryResponse(
                points=[
                    models.ScoredPoint(
                        id="p1",
                        version=0,
                        score=0.9,
                        payload={
                            "source_doc": "doc.txt",
                            "chunk_id": 1,
                            "text": "evidence text",
                        },
                        vector=None,
                    )
                ]
            )
            return response, response

        def close(self) -> None:
            self.closed = True

    pool = retriever.CorpusPool()
    monkeypatch.setattr(retriever, "_corpus_pool", pool)
    client1, client2, client3 = FakeClient(), FakeClient(), FakeClient()
    clients = iter((client1, client2, client3))
    monkeypatch.setattr(retriever, "connect_corpus", lambda: next(clients))
    queued = threading.Event()
    queued_thread: threading.Thread | None = None
    wait = pool._condition.wait

    def watch_wait(timeout: float | None = None) -> bool:
        if threading.current_thread() is queued_thread:
            queued.set()
        return wait(timeout)

    def queued_search() -> list[EvidenceItem]:
        nonlocal queued_thread
        queued_thread = threading.current_thread()
        return retriever.search_corpus("queued query")

    monkeypatch.setattr(pool._condition, "wait", watch_wait)
    with ThreadPoolExecutor(max_workers=4) as workers:
        first = workers.submit(retriever.search_corpus, "query one")
        second = workers.submit(retriever.search_corpus, "query two")
        try:
            assert client1.entered.wait(timeout=2)
            assert client2.entered.wait(timeout=2)
            pending = workers.submit(queued_search)
            assert queued.wait(timeout=2)
            shutdown = workers.submit(retriever.close_corpus)
            with pytest.raises(RuntimeError, match="shutting down"):
                pending.result(timeout=2)
            assert not client1.closed and not client2.closed
        finally:
            client1.release.set()
            client2.release.set()

        assert len(first.result(timeout=2)) == len(second.result(timeout=2)) == 1
        shutdown.result(timeout=2)

    assert client1.closed and client2.closed
    client3.release.set()
    assert len(retriever.search_corpus("next lifespan")) == 1
    assert client3.entered.is_set()
    retriever.close_corpus()
    assert client3.closed


def test_distinct_active_models_produce_distinct_cache_outcomes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = EvidenceItem(id="corpus-1", source_doc="policy.txt", text=EVIDENCE)
    monkeypatch.setattr(settings, "qdrant_host", "unused.test")
    monkeypatch.setattr(vllm_service, "base_url", "http://unused.test/v1")
    monkeypatch.setattr(retriever, "search_corpus", lambda _query: [evidence])
    calls = install_model(
        monkeypatch, extraction=[fact("corpus-1")], citation="corpus-1"
    )

    # 1. Audit with model-a
    monkeypatch.setattr(vllm_service, "model", "model-a")
    res1 = client.post("/api/v1/audit", json={"query": QUERY}).json()
    assert not res1["cache_hit"]
    assert len(calls) == 3

    # 2. Second audit with model-a hits the cache
    res2 = client.post("/api/v1/audit", json={"query": QUERY}).json()
    assert res2["cache_hit"]
    assert len(calls) == 3

    # 3. Audit with model-b on the same query/corpus produces a cache miss
    monkeypatch.setattr(vllm_service, "model", "model-b")
    res3 = client.post("/api/v1/audit", json={"query": QUERY}).json()
    assert not res3["cache_hit"]
    assert len(calls) == 6

    # 4. Second audit with model-b hits the cache
    res4 = client.post("/api/v1/audit", json={"query": QUERY}).json()
    assert res4["cache_hit"]
    assert len(calls) == 6

    # 5. Switching back to model-a hits the model-a cached entry
    monkeypatch.setattr(vllm_service, "model", "model-a")
    res5 = client.post("/api/v1/audit", json={"query": QUERY}).json()
    assert res5["cache_hit"]
    assert len(calls) == 6

    # 6. Verify key generation includes active model and v4 prefix
    key_a = cache_module.cache_service.key(
        QUERY, True, vllm_api_base="http://unused.test/v1", vllm_model="model-a"
    )
    key_b = cache_module.cache_service.key(
        QUERY, True, vllm_api_base="http://unused.test/v1", vllm_model="model-b"
    )
    assert key_a != key_b
    assert key_a.startswith("agentsentry:audit:v4:")


@pytest.mark.parametrize("strict", [True, False])
def test_original_assertion_payload_strict_and_relaxed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    original = "Production logs are stored in US and retained for 30 days."
    c1 = "Production logs are stored in US."
    c2 = "Production logs are retained for 30 days."

    chunk_anchored = "Production logs are stored in US data centers."
    chunk_unanchored = "Access to the data center is monitored 24/7."

    assertion_payloads: list[dict[str, Any]] = []

    async def generate(messages: list[dict[str, str]]) -> str:
        prompt, user = (item["content"] for item in messages)
        if prompt.startswith("Decompose"):
            return json.dumps(
                {
                    "claims": [
                        {"claim_id": "c1", "text": c1},
                        {"claim_id": "c2", "text": c2},
                    ]
                }
            )
        payload = json.loads(user)
        if prompt.startswith(verifier.EXTRACTION_PROMPT):
            if payload["claim"] == c1:
                return json.dumps(
                    {
                        "facts": [
                            {
                                "entity": "production logs",
                                "relation": "storage location",
                                "value": "US data centers",
                                "evidence_id": "context-1",
                                "quote": chunk_anchored,
                            }
                        ]
                    }
                )
            return json.dumps({"facts": []})

        assert prompt.startswith(verifier.DECISION_PROMPT)
        if payload["claim"] == original:
            assertion_payloads.append(payload)
            return json.dumps(
                {
                    "verdict": "SUPPORTED",
                    "confidence": 0.95,
                    "citation_ids": ["context-1"],
                    "rationale": "Decided against supplied facts and evidence.",
                }
            )

        is_c1 = payload["claim"] == c1
        return json.dumps(
            {
                "verdict": "SUPPORTED" if is_c1 else "INSUFFICIENT_EVIDENCE",
                "confidence": 0.95 if is_c1 else 0.0,
                "citation_ids": ["context-1"] if is_c1 else [],
                "rationale": "Claim decision.",
            }
        )

    monkeypatch.setattr(vllm_service, "generate", generate)

    result = client.post(
        "/api/v1/audit",
        json={
            "query": original,
            "context_documents": [chunk_anchored, chunk_unanchored],
            "strict_evidence_lock": strict,
        },
    ).json()

    assert len(assertion_payloads) == 1
    assertion_payload = assertion_payloads[0]

    assert assertion_payload["claim"] == original
    assert len(assertion_payload["frozen_facts"]) == 1
    assert assertion_payload["frozen_facts"][0]["evidence_id"] == "context-1"

    if strict:
        assert "evidence" not in assertion_payload
    else:
        assert "evidence" in assertion_payload
        evidence_ids = [item["id"] for item in assertion_payload["evidence"]]
        assert evidence_ids == ["context-1", "context-2"]

    assert result["status"] == "SUPPORTED"
    assert result["verdict"]["assertion_verification"]["citation_ids"] == ["context-1"]


def test_original_assertion_relaxed_mode_rejects_unanchored_citation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = "Production logs are stored in US and retained for 30 days."
    c1 = "Production logs are stored in US."
    c2 = "Production logs are retained for 30 days."
    chunk_anchored = "Production logs are stored in US data centers."
    chunk_unanchored = "Access to the data center is monitored 24/7."

    async def generate(messages: list[dict[str, str]]) -> str:
        prompt, user = (item["content"] for item in messages)
        if prompt.startswith("Decompose"):
            return json.dumps(
                {
                    "claims": [
                        {"claim_id": "c1", "text": c1},
                        {"claim_id": "c2", "text": c2},
                    ]
                }
            )
        payload = json.loads(user)
        if prompt.startswith(verifier.EXTRACTION_PROMPT):
            if payload["claim"] == c1:
                return json.dumps(
                    {
                        "facts": [
                            {
                                "entity": "production logs",
                                "relation": "storage location",
                                "value": "US data centers",
                                "evidence_id": "context-1",
                                "quote": chunk_anchored,
                            }
                        ]
                    }
                )
            return json.dumps({"facts": []})

        assert prompt.startswith(verifier.DECISION_PROMPT)
        if payload["claim"] == original:
            return json.dumps(
                {
                    "verdict": "SUPPORTED",
                    "confidence": 0.95,
                    "citation_ids": ["context-2"],
                    "rationale": "Attempting to cite unanchored chunk.",
                }
            )

        is_c1 = payload["claim"] == c1
        return json.dumps(
            {
                "verdict": "SUPPORTED" if is_c1 else "INSUFFICIENT_EVIDENCE",
                "confidence": 0.95 if is_c1 else 0.0,
                "citation_ids": ["context-1"] if is_c1 else [],
                "rationale": "Claim decision.",
            }
        )

    monkeypatch.setattr(vllm_service, "generate", generate)

    result = client.post(
        "/api/v1/audit",
        json={
            "query": original,
            "context_documents": [chunk_anchored, chunk_unanchored],
            "strict_evidence_lock": False,
        },
    ).json()

    assertion = result["verdict"]["assertion_verification"]
    assert assertion["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert assertion["issue"] == "INVALID_MODEL_OUTPUT"
    assert assertion["citation_ids"] == []

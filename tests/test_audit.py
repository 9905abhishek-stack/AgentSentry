import asyncio
import hashlib
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.graph.nodes import retriever, verifier
from app.graph.nodes.planner import order_claims
from app.schemas.agent_state import AuditResponse, Claim, EvidenceItem
from app.services import cache as cache_module
from app.services.cache import CacheService
from app.services.vllm_client import VLLMClient, vllm_service

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
    citation: str = "context-1",
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
                "citation_ids": [citation],
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
    assert [stage for stage, _ in calls] == ["plan", "extract", "decide"]


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

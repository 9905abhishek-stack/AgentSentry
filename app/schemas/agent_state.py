"""API and workflow models for claims, evidence, and audit results."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

MAX_CLAIMS = 10
MAX_CONTEXT_CHUNKS = 8
EvidenceText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]
QueryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=5, max_length=4000)
]
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class VerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class EvidenceChunk(BaseModel):
    text: EvidenceText
    source_doc: NonBlank
    chunk_id: int = Field(ge=0)


class EvidenceItem(EvidenceChunk):
    id: NonBlank
    chunk_id: int = Field(default=0, ge=0)
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_rrf_score: float = 0.0


class Claim(BaseModel):
    claim_id: NonBlank
    text: Annotated[NonBlank, Field(max_length=4000)]
    category: str = "factual"
    dependencies: list[NonBlank] = Field(default_factory=list, max_length=MAX_CLAIMS)


class FrozenFact(BaseModel, frozen=True):
    entity: NonBlank
    relation: NonBlank
    value: NonBlank
    evidence_id: NonBlank
    quote: NonBlank
    evidence_sha256: str


class VerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    verdict: VerdictEnum
    confidence: float = Field(ge=0, le=1)
    citation_ids: list[str] = Field(default_factory=list)
    rationale: str
    frozen_ir_hash: str | None = None
    frozen_ir: tuple[FrozenFact, ...] = ()


class AuditVerdict(BaseModel):
    overall_status: VerdictEnum
    verified_claims: list[VerificationResult] = Field(default_factory=list)
    aggregated_confidence: float = Field(ge=0, le=1)
    requires_hitl: bool = False
    audit_summary: str
    total_latency_ms: float = 0.0


class AgentState(BaseModel):
    query: str
    strict_evidence_lock: bool = True
    context_documents: list[EvidenceText] | None = Field(
        default=None, max_length=MAX_CONTEXT_CHUNKS
    )
    claims: list[Claim] = Field(default_factory=list)
    current_claim_index: int = 0
    retrieved_evidence: dict[str, list[EvidenceItem]] = Field(default_factory=dict)
    verifications: list[VerificationResult] = Field(default_factory=list)
    requires_human_review: bool = False
    audit_verdict: AuditVerdict | None = None
    execution_log: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: QueryText
    context_documents: list[EvidenceText] | None = Field(
        default=None,
        max_length=MAX_CONTEXT_CHUNKS,
        description="Evidence chunks supplied for this request in place of the configured corpus.",
    )
    strict_evidence_lock: bool | None = Field(
        default=None,
        description="Restrict the decision prompt to frozen facts. Defaults to the server setting.",
    )


class AuditResponse(BaseModel):
    query: str
    status: VerdictEnum
    verdict: AuditVerdict
    evidence_items: list[EvidenceItem]
    latency_ms: float
    cache_hit: bool = False

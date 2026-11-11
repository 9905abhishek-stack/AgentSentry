"""
AgentSentry Schemas & LangGraph State Definition
"""
from typing import List, Dict, Optional, Annotated
from enum import Enum
from pydantic import BaseModel, Field
import operator


class VerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class EvidenceItem(BaseModel):
    id: str = Field(..., description="Unique hash or ID of the evidence chunk")
    text: str = Field(..., description="Extracted textual content of the chunk")
    source_doc: str = Field(..., description="Origin document filename or URI")
    chunk_id: int = Field(default=0, description="Sequential chunk index in source document")
    dense_score: float = Field(default=0.0, description="Cosine similarity from dense embedding (BGE-M3)")
    sparse_score: float = Field(default=0.0, description="Lexical matching score from BM25")
    fused_rrf_score: float = Field(default=0.0, description="Combined Reciprocal Rank Fusion score")


class Claim(BaseModel):
    claim_id: str = Field(..., description="Unique identifier for the sub-claim")
    text: str = Field(..., description="Atomic verifiable statement decomposed from the input")
    category: str = Field(default="factual", description="Category: factual, numerical, temporal, privacy")
    dependencies: List[str] = Field(default_factory=list, description="IDs of prior claims this depends on")


class VerificationResult(BaseModel):
    claim_id: str = Field(..., description="Reference ID of the claim verified")
    verdict: VerdictEnum = Field(..., description="Evidence-locked verdict outcome")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score [0.0 - 1.0]")
    citation_ids: List[str] = Field(default_factory=list, description="IDs of supporting evidence chunks")
    rationale: str = Field(..., description="Deterministic chain of reasoning strictly grounded in citations")
    frozen_ir_hash: str = Field(..., description="SHA-256 hash of the frozen intermediate representation (FIR)")


class AuditVerdict(BaseModel):
    overall_status: VerdictEnum = Field(..., description="Final synthesized verdict across all claims")
    verified_claims: List[VerificationResult] = Field(default_factory=list)
    aggregated_confidence: float = Field(..., ge=0.0, le=1.0)
    requires_hitl: bool = Field(default=False, description="Flag indicating if confidence fell below SLA threshold")
    audit_summary: str = Field(..., description="Executive summary of findings with explicit citations")
    total_latency_ms: float = Field(default=0.0)


# --- LangGraph State Definition ---
class AgentState(BaseModel):
    """
    StateGraph shared state for the AgentSentry multi-agent audit DAG.
    Uses operator.add annotations where list appending is required across nodes.
    """
    query: str
    claims: List[Claim] = Field(default_factory=list)
    current_claim_index: int = 0
    retrieved_evidence: Dict[str, List[EvidenceItem]] = Field(default_factory=dict)
    frozen_ir_store: Dict[str, str] = Field(default_factory=dict)  # Claim ID -> Canonical FIR Text
    verifications: List[VerificationResult] = Field(default_factory=list)
    retry_count: int = 0
    requires_human_review: bool = False
    audit_verdict: Optional[AuditVerdict] = None
    execution_log: List[str] = Field(default_factory=list)


# --- API Request & Response Schemas ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Factual, policy, or data assertion to audit")
    context_documents: Optional[List[str]] = Field(
        default=None,
        description="Optional ad-hoc reference documents to index dynamically into Qdrant for this audit"
    )
    strict_evidence_lock: bool = Field(
        default=True,
        description="Enforces zero-hallucination FIR constraint; rejects ungrounded external claims"
    )


class AuditResponse(BaseModel):
    query: str
    status: VerdictEnum
    verdict: AuditVerdict
    evidence_items: List[EvidenceItem]
    latency_ms: float
    cache_hit: bool = False

"""Adjudicate the original assertion and flag findings for human review."""

from statistics import mean
from typing import Any

from app.config import settings
from app.graph.nodes.verifier import canonicalize_facts, decide_claim
from app.schemas.agent_state import AgentState, AuditVerdict, Claim, VerdictEnum

UNCERTAIN = {VerdictEnum.CONFLICTING_EVIDENCE, VerdictEnum.INSUFFICIENT_EVIDENCE}


async def supervise_and_synthesize(state: AgentState) -> dict[str, Any]:
    verdicts = [result.verdict for result in state.verifications]
    complete = {claim.claim_id for claim in state.claims} == {
        result.claim_id for result in state.verifications
    } and bool(verdicts)
    confidences = [result.confidence for result in state.verifications]
    confidence = mean(confidences) if confidences else 0.0
    overall = VerdictEnum.INSUFFICIENT_EVIDENCE
    assertion_verification = None
    if complete:
        if len(state.claims) == 1 and state.claims[0].text == state.query:
            overall = state.verifications[0].verdict
        else:
            facts, digest = canonicalize_facts(
                [fact for result in state.verifications for fact in result.frozen_ir]
            )
            anchored_ids = {fact.evidence_id for fact in facts}
            evidence = {
                item.id: item
                for items in state.retrieved_evidence.values()
                for item in items
                if item.id in anchored_ids
            }
            assertion_verification = await decide_claim(
                Claim(claim_id="assertion", text=state.query),
                list(evidence.values()),
                facts,
                digest,
                state.strict_evidence_lock,
            )
            overall = assertion_verification.verdict
            confidence = assertion_verification.confidence

    review = (
        state.requires_human_review
        or not complete
        or overall in UNCERTAIN
        or confidence < settings.confidence_threshold
        or any(
            result.confidence < settings.confidence_threshold
            or result.verdict in UNCERTAIN
            for result in state.verifications
        )
    )

    supported = verdicts.count(VerdictEnum.SUPPORTED)
    refuted = verdicts.count(VerdictEnum.REFUTED)
    uncertain = len(state.claims) - supported - refuted
    summary = f"{supported} supported, {refuted} refuted, {uncertain} uncertain."
    if review:
        summary += " Human review required."

    verdict = AuditVerdict(
        overall_status=overall,
        verified_claims=state.verifications,
        assertion_verification=assertion_verification,
        aggregated_confidence=round(confidence, 3),
        requires_hitl=review,
        audit_summary=summary,
    )
    return {
        "audit_verdict": verdict,
        "requires_human_review": review,
        "execution_log": [
            *state.execution_log,
            f"Audit completed with status {overall.value}.",
        ],
    }

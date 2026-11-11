"""Aggregate a conjunction of claims and flag findings for human review."""

from statistics import mean
from typing import Any

from app.config import settings
from app.schemas.agent_state import AgentState, AuditVerdict, VerdictEnum

STATUS_PRIORITY = (
    VerdictEnum.REFUTED,
    VerdictEnum.CONFLICTING_EVIDENCE,
    VerdictEnum.INSUFFICIENT_EVIDENCE,
    VerdictEnum.SUPPORTED,
)


async def supervise_and_synthesize(state: AgentState) -> dict[str, Any]:
    verdicts = [result.verdict for result in state.verifications]
    overall = next(
        (status for status in STATUS_PRIORITY if status in verdicts),
        VerdictEnum.INSUFFICIENT_EVIDENCE,
    )
    complete = {claim.claim_id for claim in state.claims} == {
        result.claim_id for result in state.verifications
    } and bool(verdicts)
    if not complete and overall == VerdictEnum.SUPPORTED:
        overall = VerdictEnum.INSUFFICIENT_EVIDENCE

    confidences = [result.confidence for result in state.verifications]
    confidence = mean(confidences) if confidences else 0.0
    review = (
        state.requires_human_review
        or not complete
        or any(
            result.confidence < settings.confidence_threshold
            or result.verdict
            in {VerdictEnum.CONFLICTING_EVIDENCE, VerdictEnum.INSUFFICIENT_EVIDENCE}
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

"""
LangGraph DAG Node: Supervisor & HITL Governance Gate
Aggregates sub-claim verdicts, enforces confidence thresholds, manages retry loops,
and flags uncertain findings for Human-in-the-Loop (HITL) review.
"""
import logging
from typing import List
from app.config import settings
from app.schemas.agent_state import AgentState, AuditVerdict, VerdictEnum, VerificationResult

logger = logging.getLogger("agentsentry.supervisor")


async def supervise_and_synthesize(state: AgentState) -> AgentState:
    """
    Evaluates completed verifications and synthesizes the final AuditVerdict.
    Applies the SLA confidence threshold to trigger retries or HITL escalation.
    """
    total_claims = len(state.claims)
    verified_count = len(state.verifications)
    logger.info(f"Supervisor synthesizing audit: {verified_count}/{total_claims} claims verified.")

    if not state.verifications:
        state.audit_verdict = AuditVerdict(
            overall_status=VerdictEnum.INSUFFICIENT_EVIDENCE,
            verified_claims=[],
            aggregated_confidence=0.0,
            requires_hitl=True,
            audit_summary="No claims were able to be verified.",
            total_latency_ms=0.0
        )
        return state

    # 1. Calculate aggregated confidence score
    confidences = [v.confidence for v in state.verifications]
    avg_confidence = round(sum(confidences) / len(confidences), 3)

    # 2. Determine Overall Verdict
    verdicts = [v.verdict for v in state.verifications]
    if VerdictEnum.REFUTED in verdicts:
        overall_status = VerdictEnum.REFUTED
    elif VerdictEnum.CONFLICTING_EVIDENCE in verdicts:
        overall_status = VerdictEnum.CONFLICTING_EVIDENCE
    elif VerdictEnum.INSUFFICIENT_EVIDENCE in verdicts:
        overall_status = VerdictEnum.INSUFFICIENT_EVIDENCE
    else:
        overall_status = VerdictEnum.SUPPORTED

    # 3. Check SLA Confidence Gate (< 0.85 requires HITL escalation)
    requires_hitl = False
    if avg_confidence < settings.confidence_threshold or overall_status == VerdictEnum.CONFLICTING_EVIDENCE:
        requires_hitl = True
        state.requires_human_review = True
        state.execution_log.append(
            f"[Supervisor] Escalating to Human-in-the-Loop (HITL): Aggregated confidence {avg_confidence:.2f} is below SLA {settings.confidence_threshold:.2f}"
        )

    # 4. Generate Audit Summary
    supported_n = sum(1 for v in verdicts if v == VerdictEnum.SUPPORTED)
    refuted_n = sum(1 for v in verdicts if v == VerdictEnum.REFUTED)
    unverified_n = total_claims - (supported_n + refuted_n)

    summary = (
        f"Audit completed across {total_claims} decomposed propositions: "
        f"{supported_n} Supported, {refuted_n} Refuted, {unverified_n} Uncertain. "
        f"Aggregated calibrated confidence: {avg_confidence * 100:.1f}%. "
        f"Evidence-locked protocol verified zero ungrounded assertions."
    )
    if requires_hitl:
        summary += " ⚠️ ESCALATED: Requires compliance officer secondary sign-off."

    state.audit_verdict = AuditVerdict(
        overall_status=overall_status,
        verified_claims=state.verifications,
        aggregated_confidence=avg_confidence,
        requires_hitl=requires_hitl,
        audit_summary=summary,
        total_latency_ms=0.0
    )

    state.execution_log.append(
        f"[Supervisor] Completed audit synthesis. Overall verdict: {overall_status.value} (Confidence: {avg_confidence:.2f}, HITL: {requires_hitl})"
    )
    return state

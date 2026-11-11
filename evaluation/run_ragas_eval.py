"""Measure fixture verdict accuracy and evidence-order stability with real inference."""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.nodes.verifier import verify_claim
from app.schemas.agent_state import Claim, EvidenceItem, VerdictEnum
from app.services.vllm_client import vllm_service


@dataclass(frozen=True)
class Fixture:
    claim: str
    evidence: tuple[str, str]
    expected: VerdictEnum


FIXTURES = (
    Fixture(
        "The payment gateway encrypts cardholder data at rest using AES-256.",
        (
            "The payment gateway encrypts stored cardholder data using AES-256.",
            "TLS is required in transit.",
        ),
        VerdictEnum.SUPPORTED,
    ),
    Fixture(
        "The access logs are deleted after 90 days.",
        (
            "The access logs are deleted after 30 days.",
            "The access logs include timestamps.",
        ),
        VerdictEnum.REFUTED,
    ),
    Fixture(
        "The payment gateway has PCI DSS Level 1 certification.",
        (
            "The payment gateway uses AES-256 at rest.",
            "The gateway uses TLS in transit.",
        ),
        VerdictEnum.INSUFFICIENT_EVIDENCE,
    ),
    Fixture(
        "Production access logs are retained for 30 days.",
        (
            "The production access logs are retained for 30 days, according to the July inventory.",
            "The same production access logs are retained for 90 days, according to a separate July inventory.",
        ),
        VerdictEnum.CONFLICTING_EVIDENCE,
    ),
)


@dataclass(frozen=True)
class EvalMetrics:
    fixture_count: int
    verdict_accuracy: float
    evidence_order_flip_rate: float
    average_verification_latency_ms: float


async def run_evaluation() -> EvalMetrics:
    if await vllm_service.health() != "available":
        raise RuntimeError(
            "Configure VLLM_API_BASE and VLLM_MODEL to run the evaluation."
        )
    correct = flips = 0
    elapsed = 0.0
    for fixture in FIXTURES:
        claim = Claim(claim_id="c1", text=fixture.claim)
        evidence = [
            EvidenceItem(id=f"e{index}", source_doc=f"fixture-{index}", text=text)
            for index, text in enumerate(fixture.evidence, 1)
        ]
        started = time.perf_counter()
        baseline = await verify_claim(claim, evidence, strict=True)
        reordered = await verify_claim(claim, list(reversed(evidence)), strict=True)
        elapsed += time.perf_counter() - started
        correct += baseline.verdict == fixture.expected
        flips += baseline.verdict != reordered.verdict

    count = len(FIXTURES)
    return EvalMetrics(
        count, correct / count, flips / count, elapsed * 1000 / (2 * count)
    )


async def evaluate_and_close() -> EvalMetrics:
    try:
        return await run_evaluation()
    finally:
        await vllm_service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        results = asyncio.run(evaluate_and_close())
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(asdict(results), indent=2))


if __name__ == "__main__":
    main()

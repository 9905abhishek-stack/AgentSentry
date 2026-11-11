"""
LangGraph DAG Node: Planner / Decomposer
Deconstructs complex user assertions into atomic, verifiable sub-claims with dependency mapping.
"""
import json
import logging
from app.schemas.agent_state import AgentState, Claim
from app.services.vllm_client import vllm_service

logger = logging.getLogger("agentsentry.planner")

PLANNER_SYSTEM_PROMPT = """You are the Lead Audit Planner in the AgentSentry verification system.
Your mission is to deconstruct complex user assertions, policies, or claims into atomic, verifiable propositions.

Rules:
1. Break down the query into distinct, non-overlapping claims.
2. Identify dependencies where claim B assumes claim A is true.
3. Classify each claim: 'factual', 'numerical', 'temporal', or 'privacy'.
4. Return strict JSON format with key "claims".

Example Output:
{
  "claims": [
    {
      "claim_id": "c1",
      "text": "Organization X holds ISO 27001 certification as of 2024.",
      "category": "factual",
      "dependencies": []
    },
    {
      "claim_id": "c2",
      "text": "Organization X encrypts customer PII at rest using AES-256.",
      "category": "privacy",
      "dependencies": ["c1"]
    }
  ]
}
"""


async def plan_and_decompose(state: AgentState) -> AgentState:
    """
    Analyzes the query and populates state.claims with atomic propositions.
    """
    logger.info(f"Decomposing query: '{state.query}'")
    state.execution_log.append(f"[Planner] Initiated query decomposition: {state.query[:60]}...")

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"DECOMPOSE this statement into atomic verifiable claims: '{state.query}'"}
    ]

    response_text = await vllm_service.generate(
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    try:
        data = json.loads(response_text)
        raw_claims = data.get("claims", [])
        parsed_claims = [Claim(**c) for c in raw_claims]
    except Exception as e:
        logger.warning(f"Failed to parse LLM decomposition response ({e}). Falling back to single atomic claim.")
        parsed_claims = [
            Claim(
                claim_id="c1",
                text=state.query.strip(),
                category="factual",
                dependencies=[]
            )
        ]

    state.claims = parsed_claims
    state.current_claim_index = 0
    state.execution_log.append(f"[Planner] Decomposed into {len(parsed_claims)} atomic claim(s).")
    return state

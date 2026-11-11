"""
LangGraph Multi-Agent StateGraph Workflow
Defines the directed acyclic graph (DAG), node edges, and conditional routing logic.
"""
import logging
from langgraph.graph import StateGraph, END
from app.schemas.agent_state import AgentState
from app.graph.nodes.planner import plan_and_decompose
from app.graph.nodes.retriever import retrieve_evidence
from app.graph.nodes.verifier import verify_evidence_locked
from app.graph.nodes.supervisor import supervise_and_synthesize

logger = logging.getLogger("agentsentry.workflow")


def should_continue_verifying(state: AgentState) -> str:
    """
    Conditional routing function:
    Determines if more sub-claims need retrieval & verification or if we proceed to supervision.
    """
    if state.current_claim_index < len(state.claims):
        return "retriever"
    return "supervisor"


def build_audit_graph():
    """
    Constructs and compiles the AgentSentry LangGraph DAG.
    """
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("planner", plan_and_decompose)
    workflow.add_node("retriever", retrieve_evidence)
    workflow.add_node("verifier", verify_evidence_locked)
    workflow.add_node("supervisor", supervise_and_synthesize)

    # 2. Register Edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "verifier")

    # 3. Conditional verification loop for multi-claim DAG
    workflow.add_conditional_edges(
        "verifier",
        should_continue_verifying,
        {
            "retriever": "retriever",
            "supervisor": "supervisor"
        }
    )

    # 4. Supervisor terminates into END
    workflow.add_edge("supervisor", END)

    # Compile the graph
    app = workflow.compile()
    return app


# Pre-compiled application instance
audit_dag = build_audit_graph()

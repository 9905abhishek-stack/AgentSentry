"""Compile the LangGraph workflow shared by REST, streaming, and evaluation."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes.planner import plan_and_decompose
from app.graph.nodes.retriever import retrieve_evidence
from app.graph.nodes.supervisor import supervise_and_synthesize
from app.graph.nodes.verifier import verify_evidence_locked
from app.schemas.agent_state import AgentState


def next_node(state: AgentState) -> Literal["retriever", "supervisor"]:
    if state.current_claim_index < len(state.claims):
        return "retriever"
    return "supervisor"


def build_audit_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", plan_and_decompose)
    workflow.add_node("retriever", retrieve_evidence)
    workflow.add_node("verifier", verify_evidence_locked)
    workflow.add_node("supervisor", supervise_and_synthesize)
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", next_node)
    workflow.add_edge("retriever", "verifier")
    workflow.add_conditional_edges("verifier", next_node)
    workflow.add_edge("supervisor", END)
    return workflow.compile()


audit_dag = build_audit_graph()

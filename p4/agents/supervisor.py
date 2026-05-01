"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""
from __future__ import annotations

from typing import Any
from langgraph.graph import END, StateGraph
from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.state import PlanTask, ResearchState, _append_scratchpad, _advance_plan

try:
    from langgraph.types import interrupt
except ImportError:
    interrupt = None

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_HITL_CONFIDENCE_THRESHOLD = 0.75

def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """

    question = state["user_question"]

    plan: list[PlanTask] = [
        {
            "id": 1,
            "task_type": "retrieve",
            "description": f"Retrieve relevant source chunks for: {question}",
            "status": "pending",
        },
        {
            "id": 2,
            "task_type": "analyze",
            "description": "Synthesize an answer using the retrieved chunks.",
            "status": "pending",
        },
        {
            "id": 3,
            "task_type": "fact_check",
            "description": "Verify the answer against trusted fact-check sources.",
            "status": "pending",
        },
    ]

    return {
        "current_plan": plan,
        "current_task_index": 0,
        "current_task": plan[0],
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "analysis_output": state.get("analysis_output", ""),
        "fact_check_results": state.get("fact_check_results", {}),
        "confidence_score": state.get("confidence_score", 0.0),
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", DEFAULT_MAX_ITERATIONS),
        "hitl_required": False,
        "scratchpad": _append_scratchpad(
            state,
            f"Planner created {len(plan)} tasks for question: {question}",
        ),
    }


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    current_task = state.get("current_task")

    if not current_task:
        return "critique"

    task_type = current_task["task_type"]

    if task_type == "retrieve":
        return "retriever"

    if task_type == "analyze":
        return "analyst"

    if task_type == "fact_check":
        return "fact_checker"

    return "critique"


def fact_checker_node(state: ResearchState) -> dict[str, Any]:
    """
    Fact-checker node placeholder.

    Replace this section with your real fact-checking agent call.
    """

    confidence = state.get("confidence_score", 0.0)

    fact_check_results = {
        "overall_verdict": "Supported" if confidence >= 0.75 else "Inconclusive",
        "unsupported_claims": [],
        "evidence": state.get("retrieved_chunks", []),
    }

    updates = _advance_plan(state, "Fact-checker")

    return {
        **updates,
        "fact_check_results": fact_check_results,
    }


def critique_node(state: ResearchState) -> dict:
    """
    Evaluate the aggregated response and decide: accept, retry, or escalate.

    TODO:
    - Check confidence_score against the HITL threshold.
    - If below threshold and iterations < max, loop back for refinement.
    - If below threshold and iterations >= max, trigger HITL interrupt.
    - If above threshold, accept and route to END.
    - Increment iteration_count.
    """
    confidence = state.get("confidence_score", 0.0)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)

    fact_check_results = state.get("fact_check_results", {})
    unsupported_claims = fact_check_results.get("unsupported_claims", [])

    next_iteration_count = iteration_count + 1

    if confidence >= DEFAULT_HITL_CONFIDENCE_THRESHOLD and not unsupported_claims:
        analysis = state.get("analysis_output", {})
        final_answer = (
            analysis.get("answer", "")
            if isinstance(analysis, dict)
            else str(analysis)
        )

        return {
            "critique_decision": "accept",
            "final_answer": final_answer,
            "iteration_count": next_iteration_count,
            "hitl_required": False,
            "scratchpad": _append_scratchpad(
                state,
                f"Critique accepted response with confidence {confidence}.",
            ),
        }

    if next_iteration_count < max_iterations:
        retry_target = "retry_retriever" if unsupported_claims else "retry_analyst"

        return {
            "critique_decision": retry_target,
            "iteration_count": next_iteration_count,
            "current_task_index": 0 if retry_target == "retry_retriever" else 1,
            "current_task": state.get("current_plan", [])[0]
            if retry_target == "retry_retriever"
            else state.get("current_plan", [])[1],
            "scratchpad": _append_scratchpad(
                state,
                f"Critique requested refinement: {retry_target}.",
            ),
        }

    if interrupt is not None:
        human_feedback = interrupt(
            {
                "reason": "Low confidence or unsupported claims after max retries.",
                "confidence_score": confidence,
                "fact_check_results": fact_check_results,
                "analysis_output": state.get("analysis_output", {}),
            }
        )

        return {
            "critique_decision": "hitl",
            "hitl_required": True,
            "final_answer": str(human_feedback),
            "iteration_count": next_iteration_count,
            "scratchpad": _append_scratchpad(
                state,
                "Critique escalated to HITL review.",
            ),
        }

    return {
        "critique_decision": "hitl",
        "hitl_required": True,
        "iteration_count": next_iteration_count,
        "scratchpad": _append_scratchpad(
            state,
            "Critique marked output for HITL review.",
        ),
    }

def critique_router(state: ResearchState) -> str:
    """
    Route after critique.
    """

    decision = state.get("critique_decision")

    if decision == "accept":
        return "end"

    if decision == "retry_retriever":
        return "retriever"

    if decision == "retry_analyst":
        return "analyst"

    if decision == "hitl":
        return "end"

    return "end"


def build_supervisor_graph():
    """
    Construct and compile the Supervisor StateGraph.

    TODO:
    - Instantiate StateGraph with ResearchState.
    - Add nodes: planner, retriever, analyst, fact_checker, critique.
    - Add edges and conditional edges (router).
    - Set entry point to planner.
    - Compile and return the graph.

    Returns:
        A compiled LangGraph that can be invoked with an initial state.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("critique", critique_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        router,
        {
            "retriever": "retriever",
            "analyst": "analyst",
            "fact_checker": "fact_checker",
            "critique": "critique",
        },
    )

    graph.add_conditional_edges(
        "retriever",
        router,
        {
            "retriever": "retriever",
            "analyst": "analyst",
            "fact_checker": "fact_checker",
            "critique": "critique",
        },
    )

    graph.add_conditional_edges(
        "analyst",
        router,
        {
            "retriever": "retriever",
            "analyst": "analyst",
            "fact_checker": "fact_checker",
            "critique": "critique",
        },
    )

    graph.add_conditional_edges(
        "fact_checker",
        router,
        {
            "retriever": "retriever",
            "analyst": "analyst",
            "fact_checker": "fact_checker",
            "critique": "critique",
        },
    )

    graph.add_conditional_edges(
        "critique",
        critique_router,
        {
            "retriever": "retriever",
            "analyst": "analyst",
            "end": END,
        },
    )

    return graph.compile()

def main():
    # NOTE: This code is temporarily placed here. It demonstrates successful integration between Pinecone retrieval and supervisor agent. Replace the user_question and see what results you get!
    app = build_supervisor_graph()

    result = app.invoke({
        "user_question": "What is utilitarianism?",
        "scratchpad": []
    })

    print("\n=== FINAL ANSWER ===")
    print(result.get("final_answer"))

    print("\n=== RETRIEVED CHUNKS ===")
    for c in result.get("retrieved_chunks", []):
        print(f"- {c['chunk_id']} (score={c['relevance_score']:.3f})")

if __name__ == "__main__":
    main()
    
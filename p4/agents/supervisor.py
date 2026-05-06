"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

from __future__ import annotations

from typing import Any, Optional

from dotenv import load_dotenv
from langgraph.checkpoint.memory import (
    MemorySaver,
)  # enables checkpoint history / time travel
from langgraph.graph import END, CompiledStateGraph, StateGraph

from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node
from agents.retriever import retriever_node
from agents.state import PlanTask, ResearchState, _append_scratchpad

load_dotenv()

try:
    from langgraph.types import Command, interrupt, StateSnapshot
    # interrupt = None
except ImportError:
    interrupt = None
    Command = None  # HITL pause/resume support
    StateSnapshot = None

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_HITL_CONFIDENCE_THRESHOLD = 0.75
CHECKPOINTER = MemorySaver()  # stores graph state after each node and enables HITL resume, view prev states, rewinding/forking from prev checkpoint  # noqa: E501


# helper for self-refinement loop
def _reset_task_for_retry(
    state: ResearchState,
    retry_target: str,
) -> dict[str, Any]:
    """
    CHANGE: Handles Plan-and-Execute self-refinement.

    - retry_retriever → restart full pipeline
    - retry_analyst → reuse retrieval, redo analysis + fact-check
    """
    plan = state.get("current_plan", [])

    for task in plan:
        if retry_target == "retry_retriever":
            task["status"] = "pending"
        elif retry_target == "retry_analyst" and task["task_type"] in {
            "analyze",
            "fact_check",
        }:
            task["status"] = "pending"

    next_index = 0 if retry_target == "retry_retriever" else 1

    return {
        "current_plan": plan,
        "current_task_index": next_index,
        "current_task": plan[next_index] if next_index < len(plan) else None,
    }


def planner_node(state: ResearchState) -> dict[str, Any]:
    """
    Decompose the user's question into actionable sub-tasks.

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
        "critique_decision": "",
        "hitl_required": False,
        "scratchpad": _append_scratchpad(
            state,
            # f"Planner created {len(plan)} tasks for question: {question}",
            f"Plan-and-Execute created {len(plan)} tasks.",
        ),
    }


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

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


def _get_final_answer_from_state(state: ResearchState) -> str:
    """
    Safely extract the best available final answer from analysis_output.

    Prevents final_answer from becoming None.
    """
    analysis = state.get("analysis_output", {})

    if isinstance(analysis, dict):
        return analysis.get("answer", "")

    return str(analysis)


def _safe_plan_task(state: ResearchState, index: int) -> list[PlanTask] | None:
    """
    Safely fetch a task from the current plan.

    Prevents index errors during HITL retries.
    """
    plan = state.get("current_plan", [])

    if index < len(plan):
        return plan[index]

    return None


def _build_hitl_payload(
    state: ResearchState,
    confidence: float,
    unsupported_claims: list[Any],
) -> dict[str, Any]:
    """Payload shown to the human reviewer when the graph pauses."""
    return {
        "reason": "Low confidence or unsupported claims after max retries.",
        "confidence_score": confidence,
        "unsupported_claims": unsupported_claims,
        "analysis_output": state.get("analysis_output", {}),
        "fact_check_results": state.get("fact_check_results", {}),
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "review_options": {
            "approve": "Accept the generated answer.",
            "revise": "Provide a corrected final_answer.",
            "retry_retriever": "Send graph back to retrieval.",
            "retry_analyst": "Send graph back to analysis.",
        },
    }


def _handle_human_feedback(
    state: ResearchState,
    human_feedback: object,
    next_iteration_count: int,
) -> dict[str, Any]:
    """Convert reviewer feedback into graph state updates."""
    if not isinstance(human_feedback, dict):
        return {
            "critique_decision": "accept",
            "hitl_required": False,
            "final_answer": _get_final_answer_from_state(state),
            "iteration_count": next_iteration_count,
            "scratchpad": _append_scratchpad(
                state,
                "Human reviewer returned plain text feedback.",
            ),
        }

    action = human_feedback.get("action", "approve")
    feedback_text = human_feedback.get("feedback", "")

    if action == "approve":
        return {
            "critique_decision": "accept",
            "hitl_required": False,
            "final_answer": _get_final_answer_from_state(state),
            "iteration_count": next_iteration_count,
            "scratchpad": _append_scratchpad(
                state,
                "Human reviewer approved the generated answer.",
            ),
        }

    if action == "revise":
        return {
            "critique_decision": "accept",
            "hitl_required": False,
            "final_answer": human_feedback.get("final_answer", ""),
            "iteration_count": next_iteration_count,
            "scratchpad": _append_scratchpad(
                state,
                "Human reviewer revised the final answer.",
            ),
        }

    if action == "retry_retriever":
        return {
            "critique_decision": "retry_retriever",
            "hitl_required": False,
            "iteration_count": next_iteration_count,
            "current_task_index": 0,
            "current_task": _safe_plan_task(state, 0),
            "scratchpad": _append_scratchpad(
                state,
                f"Human reviewer requested retriever retry. Feedback: {feedback_text}",
            ),
        }

    if action == "retry_analyst":
        return {
            "critique_decision": "retry_analyst",
            "hitl_required": False,
            "iteration_count": next_iteration_count,
            "current_task_index": 1,
            "current_task": _safe_plan_task(state, 1),
            "scratchpad": _append_scratchpad(
                state,
                f"Human reviewer requested analyst retry. Feedback: {feedback_text}",
            ),
        }

    return {
        "critique_decision": "hitl",
        "hitl_required": True,
        "iteration_count": next_iteration_count,
        "final_answer": _get_final_answer_from_state(state),
        "scratchpad": _append_scratchpad(
            state,
            f"Unknown HITL action received: {action}",
        ),
    }


def critique_node(state: ResearchState) -> dict[str, Any]:
    """
    Evaluate the aggregated response and decide: accept, retry, or escalate.

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
        # analysis = state.get("analysis_output", {})
        # final_answer = (analysis.get("answer", "") if isinstance(analysis, dict) else str(analysis))  # noqa: E501

        return {
            "critique_decision": "accept",
            # "final_answer": final_answer,
            "final_answer": _get_final_answer_from_state(state),
            "iteration_count": next_iteration_count,
            "hitl_required": False,
            "scratchpad": _append_scratchpad(
                state,
                f"Critique accepted response with confidence {confidence}.",
            ),
        }

    # self refinement loop
    if next_iteration_count < max_iterations:
        retry_target = "retry_retriever" if unsupported_claims else "retry_analyst"

        # smarter retry selection
        retry_updates = _reset_task_for_retry(state, retry_target)

        return {
            **retry_updates,
            "critique_decision": retry_target,
            "iteration_count": next_iteration_count,
            "current_task_index": 0 if retry_target == "retry_retriever" else 1,
            "current_task": state.get("current_plan", [])[0]
            if retry_target == "retry_retriever"
            else state.get("current_plan", [])[1],
            "scratchpad": _append_scratchpad(
                state,
                f"Self-refinement triggered → {retry_target} "
                f"(confidence={confidence}, unsupported={len(unsupported_claims)})",
                # f"Critique requested refinement: {retry_target}.",
            ),
        }

    # HITL escalation after retries are exhausted.
    if interrupt is not None:
        payload = _build_hitl_payload(
            state=state,
            confidence=confidence,
            unsupported_claims=unsupported_claims,
        )

        human_feedback = interrupt(payload)

        return _handle_human_feedback(
            state=state,
            human_feedback=human_feedback,
            next_iteration_count=next_iteration_count,
        )

    # fallback when interrupt is unavailable.
    return {
        "critique_decision": "hitl",
        "hitl_required": True,
        "final_answer": _get_final_answer_from_state(state),
        "iteration_count": next_iteration_count,
        "scratchpad": _append_scratchpad(
            state,
            "Critique marked output for HITL review, but interrupt is unavailable.",
        ),
    }


def critique_router(state: ResearchState) -> str:
    """Route after critique."""
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


def build_supervisor_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """
    Construct and compile the Supervisor StateGraph.

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

    return graph.compile(checkpointer=checkpointer or CHECKPOINTER)


# time travel helpers
def build_thread_config(thread_id: str) -> dict[str, Any]:
    """Every graph run that needs checkpointing must use a thread_id."""
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def resume_from_hitl(
    app: CompiledStateGraph,
    thread_id: str,
    reviewer_feedback: dict[str, Any],
) -> dict[str, Any]:
    """Resume the graph after interrupt() pauses execution."""
    if Command is None:
        raise RuntimeError("Command is unavailable. Upgrade langgraph.")

    return app.invoke(
        Command(resume=reviewer_feedback),
        config=build_thread_config(thread_id),
    )


def get_thread_history(app: CompiledStateGraph, thread_id: str) -> list[Any]:
    """View all saved checkpoints for a graph thread."""
    return list(app.get_state_history(build_thread_config(thread_id)))


def get_latest_thread_state(app: CompiledStateGraph, thread_id: str) -> StateSnapshot:
    """Get the latest checkpointed state."""
    return app.get_state(build_thread_config(thread_id))


def fork_from_checkpoint(
    app: CompiledStateGraph,
    checkpoint_config: dict[str, Any],
    state_updates: dict[str, Any],
    as_node: Optional[str] = None,
) -> dict[str, Any]:
    """
    Time travel.

    Rewind to a previous checkpoint, apply updates, then continue execution.
    """
    if as_node:
        fork_config = app.update_state(
            checkpoint_config,
            values=state_updates,
            as_node=as_node,
        )
    else:
        fork_config = app.update_state(
            checkpoint_config,
            values=state_updates,
        )

    return app.invoke(None, config=fork_config)


def main() -> None:
    """
    NOTE: This code is temporarily placed here.

    It demonstrates successful integration between Pinecone retrieval
    and supervisor agent.
    Replace the user_question and see what results you get!
    """
    app = build_supervisor_graph()

    # to test time travel and HITL
    thread_id = "demo-thread"
    config = build_thread_config(thread_id)

    result = app.invoke(
        {
            "user_question": "What is utilitarianism?",
            "scratchpad": [],
            "iteration_count": 0,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
        },
        config=config,
    )

    if result.get("hitl_required"):
        print("\n=== HITL REQUIRED ===")

        # simulate human approval
        result = resume_from_hitl(
            app=app,
            thread_id=thread_id,
            reviewer_feedback={
                "action": "approve",
            },
        )

    print("\n=== FINAL ANSWER ===")
    print(result.get("final_answer"))

    print("\n=== RETRIEVED CHUNKS ===")
    for c in result.get("retrieved_chunks", []):
        score = c.get("relevance_score", c.get("score", 0.0))
        print(f"- {c.get('chunk_id')} (score={score:.3f})")

    print("\n=== CHECKPOINT HISTORY ===")
    history = get_thread_history(app, thread_id)
    for i, checkpoint in enumerate(history):
        print(f"{i}: next={checkpoint.next}")


# HITL test/example:
# result = resume_from_hitl(
#   app=app,
#  thread_id="demo-thread",
# reviewer_feedback={
#    "action": "revise",
#   "final_answer": "Human-approved corrected answer.",
# },
# )

if __name__ == "__main__":
    main()

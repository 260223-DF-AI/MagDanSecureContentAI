"""
ResearchFlow — Graph State Definition

Defines the TypedDict that flows through the Supervisor StateGraph.
All nodes read from and write to this shared state.
"""

from typing import Any, Literal, TypedDict

TaskType = Literal["retrieve", "analyze", "fact_check"]


class PlanTask(TypedDict):
    """Structure of an agent task."""

    id: int
    task_type: TaskType
    description: str
    status: Literal["pending", "complete"]


class ResearchState(TypedDict):
    """
    Shared state for the Supervisor graph.

    TODO: Expand these fields as your design evolves.

    Attributes:
        question: The original user research question.
        plan: Decomposed sub-tasks from the Planner node.
        retrieved_chunks: Chunks returned by the Retriever agent.
        analysis: Synthesized response from the Analyst agent.
        fact_check_report: Verification report from the Fact-Checker agent.
        confidence_score: Overall confidence in the final answer (0.0–1.0).
        iteration_count: Number of self-refinement loops executed so far.
        scratchpad: Step-wise log of intermediate outputs for observability.
        user_id: Identifier for cross-thread memory via the Store interface.
    """

    # question: str
    # plan: list[str]
    # retrieved_chunks: list[dict]
    # analysis: dict
    fact_check_report: dict
    confidence_score: float
    iteration_count: int
    scratchpad: list[str]
    user_id: str

    user_question: str
    current_plan: list[PlanTask]
    current_task_index: int
    current_task: PlanTask | None

    retrieved_chunks: list[dict[str, Any]]
    analysis_output: dict[str, Any] | str
    fact_check_results: dict[str, Any]

    confidence_score: float
    iteration_count: int
    max_iterations: int

    critique_decision: Literal["accept", "retry_retriever", "retry_analyst", "hitl"]
    final_answer: str
    hitl_required: bool

    scratchpad: list[str]


def _append_scratchpad(state: ResearchState, message: str) -> list[str]:
    return [*state.get("scratchpad", []), message]


def _advance_plan(state: ResearchState, node_name: str) -> dict[str, Any]:
    plan = state.get("current_plan", [])
    index = state.get("current_task_index", 0)
    next_task, next_index = (None, None)

    if isinstance(plan, list) and index < len(plan):
        task = plan[index]
        if isinstance(task, dict):
            task["status"] = "complete"

        next_index = index + 1
        next_task = plan[next_index] if next_index < len(plan) else None

    return {
        "current_plan": plan,
        "current_task_index": next_index if next_index is not None else 0,
        "current_task": next_task if next_task is not None else "",
        "scratchpad": _append_scratchpad(
            state,
            f"{node_name} completed task index {index}.",
        ),
    }

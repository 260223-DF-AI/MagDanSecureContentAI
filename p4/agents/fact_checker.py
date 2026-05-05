"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""

from typing import Any
from pydantic import BaseModel

from agents.state import PlanTask, ResearchState, _append_scratchpad, _advance_plan
from agents.state import ResearchState


class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""

    claim: str
    verdict: str  # "Supported" | "Unsupported" | "Inconclusive"
    evidence: str | None = None


class FactCheckReport(BaseModel):
    """Full verification report across all claims."""

    verdicts: list[ClaimVerdict]
    overall_confidence: float


def fact_checker_node(state: ResearchState) -> dict:
    """
    Verify the Analyst's response against trusted reference sources.

    TODO:
    - Extract claims from state["analysis"].
    - Query the 'fact-check-sources' Pinecone namespace for each claim.
    - Produce per-claim verdicts.
    - If confidence < threshold, trigger HITL interrupt.
    - Support Time Travel via state checkpointing.

    """
    raise NotImplementedError


def fact_checker_node(state: ResearchState) -> dict[str, Any]:
    """
    Verify the Analyst's response against trusted reference sources.

    TODO:
    - CURRENTLY COPIED CODE FROM SUPERVISOR SKELETON
    - Extract claims from state["analysis"].
    - Query the 'fact-check-sources' Pinecone namespace for each claim.
    - Produce per-claim verdicts.
    - If confidence < threshold, trigger HITL interrupt.
    - Support Time Travel via state checkpointing.

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

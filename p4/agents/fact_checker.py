"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""

import logging
import re
from typing import Any

from infrastructure.instances import _get_embedder, _get_index
from langchain_core.prompts import ChatPromptTemplate
from logs.log_config import setup_logging
from pydantic import BaseModel, Field

from agents.state import ResearchState, _advance_plan, _append_scratchpad

setup_logging()
logger = logging.getLogger("researchflow.fact_checker")

_embedder = _get_embedder()
_pinecone_index = _get_index()
_verdict_llm = None


class ClaimVerdict(BaseModel):
    """Schema for a claim-verdict object."""
    claim: str
    verdict: str = Field(pattern=r"^(Supported|Unsupported|Inconclusive)$")
    evidence: str


class FactCheckReport(BaseModel):
    """Schema for the fack checker agent's report."""
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)


_VERDICT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict fact-checker. Given a claim and supporting evidence, "
            "decide one of: Supported, Unsupported, Inconclusive.\n"
            "  • Supported = the evidence directly states or strongly implies the claim.\n"
            "  • Unsupported = the evidence contradicts the claim.\n"
            "  • Inconclusive = the evidence is silent on the claim.\n"
            "Quote a short snippet from the evidence as your justification.\n\n"
            "Output schema: return JSON with 'verdict' (one of the three labels above, "
            "exactly as spelled) and 'evidence' (a short string snippet from the input).",
        ),
        ("human", "Claim: {claim}\n\nEvidence:\n{evidence}"),
    ]
)


class _SingleVerdict(BaseModel):
    """Schema the verdict-LLM is forced into."""

    verdict: str = Field(
        pattern=r"^(Supported|Unsupported|Inconclusive)$",
        description="Exactly one of: Supported, Unsupported, Inconclusive",
    )
    evidence: str = Field(
        description="A short quoted snippet from the evidence justifying the verdict",
    )


def _split_into_claims(answer: str) -> list[str]:
    """Heuristic claim extraction — split on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s for s in sentences if len(s) > 20]


def _verify_claim(claim: str) -> ClaimVerdict:
    query_vec = _embedder.embed_query(claim)
    raw = _pinecone_index.query(
        vector=query_vec,
        top_k=3,
        namespace="fact-check-sources",
        include_metadata=True,
    )
    matches = raw.get("matches", []) if isinstance(raw, dict) else raw["matches"]
    if not matches:
        return ClaimVerdict(
            claim=claim,
            verdict="Inconclusive",
            evidence="No supporting documents found.",
        )

    evidence_block = "\n\n---\n\n".join(
        m["metadata"].get("content", "") for m in matches
    )
    chain = _VERDICT_PROMPT | _verdict_llm.with_structured_output(_SingleVerdict)
    out: _SingleVerdict = chain.invoke({"claim": claim, "evidence": evidence_block})
    return ClaimVerdict(claim=claim, verdict=out.verdict, evidence=out.evidence)


def fact_checker_node(state: ResearchState) -> dict[str, Any]:
    """
    Verify the Analyst's response against trusted reference sources.

    TODO:
    - Extract claims from state["analysis"].
    - Query the 'fact-check-sources' Pinecone namespace for each claim.
    - Produce per-claim verdicts.
    - If confidence < threshold, trigger HITL interrupt.
    - Support Time Travel via state checkpointing.

    """
    analyst_output = state.get("analysis_output", "")
    claims = _split_into_claims(analyst_output)
    logger.info(f"[Fact Checker] extracted {len(claims)} claims from analyst output.")

    if not claims:
        report = FactCheckReport(verdicts=[], overall_confidence=0.0)
        return {
            "fact_check_results": report.model_dump(),
            "confidence_score": 0.0,
            "hitl_required": True,
            "scratchpad": ["[Fact Checker] no claims found, escalating to HITL"],
        }

    verdicts = [_verify_claim(c) for c in claims]

    confidence = state.get("confidence_score", 0.0)

    fact_check_results = {
        "overall_verdict": "Supported" if confidence >= 0.75 else "Inconclusive",
        "unsupported_claims": [],
        "evidence": state.get("retrieved_chunks", []),
    }

    updates = _advance_plan(state, "fact-checker")

    return {
        **updates,
        "fact_check_results": fact_check_results,
        "hitl_required": needs_hitl,
        "scratchpad": ["[Fact Checker] no claims found, escalating to HITL"],
    }

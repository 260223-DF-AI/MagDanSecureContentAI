"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""

import logging
import os
import re
from collections import Counter
from typing import Any

from infrastructure.instances import _get_embedder, _get_index, _get_llm
from langchain_core.prompts import ChatPromptTemplate
from logs.log_config import setup_logging
from pydantic import BaseModel, Field

from agents.state import ResearchState, _advance_plan, _append_scratchpad

setup_logging()
logger = logging.getLogger("researchflow.fact_checker")

_embedder = _get_embedder()
_pinecone_index = _get_index()
_verdict_llm = _get_llm()


class ClaimVerdict(BaseModel):
    """Schema for a claim-verdict object."""

    claim: str
    verdict: str = Field(pattern=r"^(Supported|Unsupported|Inconclusive)$")
    evidence: str


class FactCheckReport(BaseModel):
    """Schema for the fack checker agent's report."""

    overall_verdict: str = Field(pattern=r"^(Supported|Unsupported|Inconclusive)$")
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str]
    evidence: list[str]


_VERDICT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise but decisive fact-checker. Given a claim and supporting"
            "evidence, decide one of: Supported, Unsupported, Inconclusive.\n"
            "  • Supported = the evidence directly states or strongly implies the claim"
            ", even if wording differs.\n"
            "  • Unsupported = the evidence contradicts the claim or states the "
            "opposite.\n"
            "  • Inconclusive = the evidence is unrelated, insufficient, or silent on "
            "the claim.\n"
            "Quote a short snippet from the evidence as your justification. \n"
            "Only choose Inconclusive when the evidence truly provides no basis for a "
            "decision.\n\n"
            "Output schema: return JSON with 'verdict' (one of the three labels above, "
            "exactly as spelled) and 'evidence' (a short snippet from the input)",
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
    """
    Use an LLM to convert the analyst's paragraph into 3–5 retrieval-optimized claims.

    These are short, keyword-rich, and designed specifically for vector search.
    """
    prompt = f"""
    You will receive an explanation written by an analyst.
    Rewrite it into 3–5 short, standalone factual claims that are
    optimized for information retrieval.

    Requirements:
    - Each claim must be 8–20 words.
    - Each claim must contain the key subject explicitly (e.g., "utilitarianism").
    - Avoid pronouns like "it", "this theory", "they".
    - Include the most important keywords needed to retrieve evidence.
    - Do NOT add new facts.

    Analyst explanation:
    {answer}

    Return the claims as a numbered list.
    """

    # Call the LLM
    response = _verdict_llm.invoke(prompt)

    # Extract text from AIMessage
    text = response.content if hasattr(response, "content") else str(response)

    # Split into lines
    lines = text.split("\n")

    claims = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove "1. ", "2) ", "- " etc.
        line = re.sub(r"^\W*\d+\W*", "", line)
        if len(line) > 5:
            claims.append(line)

    return claims


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

    evidence_block = "\n\n---\n\n".join(m["metadata"].get("text", "") for m in matches)

    chain = _VERDICT_PROMPT | _verdict_llm.with_structured_output(_SingleVerdict)
    out: _SingleVerdict = chain.invoke({"claim": claim, "evidence": evidence_block})
    logger.info(
        f"[Fact Checker] Verifying claim '{claim}' | Evidence: {evidence_block}"
    )
    return ClaimVerdict(claim=claim, verdict=out.verdict, evidence=out.evidence)


def _calc_confidence(verdicts: list, counts: dict) -> float:
    """Calculate fact checker's confidence score for all claims."""
    # If everything is inconclusive -> neutral confidence
    if counts["Supported"] == 0 and counts["Unsupported"] == 0:
        return 0.5

    # Confidence = (supported - unsupported) / total, clamped to [0, 1].
    total = max(len(verdicts), 1)
    raw = (
        counts["Supported"] - counts["Unsupported"] - 0.5 * counts["Inconclusive"]
    ) / total

    return max(0.0, min(1.0, raw))


def fact_checker_node(state: ResearchState) -> dict[str, Any]:
    """
    Verify the Analyst's response against trusted reference sources.

    - Extract claims from state["analysis"].
    - Query the 'fact-check-sources' Pinecone namespace for each claim.
    - Produce per-claim verdicts.
    - If confidence < threshold, trigger HITL interrupt.
    - Support Time Travel via state checkpointing.

    """
    analyst_output = state.get("analysis_output", "")
    answer = analyst_output["answer"]
    logger.info(f"[Fact Checker] validating Analyst's output: '{answer[:75]}...'")

    claims = _split_into_claims(analyst_output["answer"])
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
    counts = Counter(v.verdict for v in verdicts)
    confidence = _calc_confidence(verdicts, counts)
    conf_threshold = float(os.getenv("HITL_CONFIDENCE_THRESHOLD"))
    needs_hitl = confidence < conf_threshold or counts["Unsupported"] > 0
    logger.info(
        f"[Fact Checker] verified claims: {counts} | HITL required: {needs_hitl}"
    )

    unsupported = []
    evidence = []
    for v in verdicts:
        if v.verdict == "Unsupported":
            unsupported.append(v.claim)
        evidence.extend(v.evidence)

    if unsupported:
        logger.info(f"[Fact Checker] unsupported claims: {unsupported}")
    # find overall verdict
    if counts["Unsupported"] > 0:
        overall = "Unsupported"
    elif confidence >= 0.75:
        overall = "Supported"
    else:
        overall = "Inconclusive"

    results = FactCheckReport(
        overall_verdict=overall,
        overall_confidence=confidence,
        verdicts=verdicts,
        unsupported_claims=unsupported,
        evidence=evidence,
    )

    # Log to scratchpad
    scratch_msg = (
        f"[fact_checker] supported={counts['Supported']}, "
        f"unsupported={counts['Unsupported']}, inconclusive={counts['Inconclusive']}, "
        f"overall={confidence:.2f}, hitl={needs_hitl}"
    )

    updates = _append_scratchpad(state, scratch_msg)
    plan_updates = _advance_plan(state, "fact-checker")
    logger.info("Advancing plan...")

    return {
        **plan_updates,
        "fact_check_results": results.model_dump(),
        "hitl_required": needs_hitl,
        "confidence_score": confidence,
        "scratchpad": updates,
    }

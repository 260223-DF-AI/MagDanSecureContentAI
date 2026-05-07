"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from logs.log_config import setup_logging
from pydantic import BaseModel, ValidationError

from agents.state import ResearchState, _advance_plan, _append_scratchpad

setup_logging()
logger = logging.getLogger("researchflow.analyst")
# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single supporting citation."""

    source: str
    page_number: Optional[int] = None
    excerpt: str


class AnalysisResult(BaseModel):
    """Pydantic model enforcing structured analyst output."""

    answer: str
    citations: List[Citation]
    confidence: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Convert retrieved chunks into a prompt-friendly string."""
    if not chunks:
        return "No retrieved context was provided."

    formatted = []

    for i, chunk in enumerate(chunks, start=1):
        text = (
            chunk.get("text") or chunk.get("content") or chunk.get("page_content") or ""
        )

        formatted.append(
            f"[Chunk {i}]\n"
            f"source: {chunk.get('source')}\n"
            f"page_number: {chunk.get('page', 'null')}\n"
            f"{text}\n"
        )

    return "\n".join(formatted)


def _build_prompt(question: str, task: str, context: str) -> str:
    """Build the LLM prompt enforcing structured JSON output."""
    return f"""
You are an expert research analyst.

QUESTION:
{question}

TASK:
{task}

CONTEXT:
{context}

Return ONLY valid JSON.

NO markdown.
NO explanations.
NO code fences.

FORMAT:
{{
  "answer": "...",
  "citations": [
    {{
      "source": "...",
      "page_number": null,
      "excerpt": "..."
    }}
  ],
  "confidence": 0.85
}}
"""


# normalized text extraction for various chunk formats into plain string
def _extract_text(content: str | list) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(getattr(item, "text", ""))
        return "\n".join(parts)

    return str(content)


def _call_bedrock(prompt: str) -> str:
    """Invoke Bedrock Claude model."""
    llm = ChatBedrockConverse(
        model=os.getenv("BEDROCK_MODEL_ID"),
        temperature=0,
        region_name=os.getenv("AWS_REGION"),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return _extract_text(response.content)


# strips markdown and extracts JSON from mixed output
def _extract_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()

    # remove ```json blocks
    raw = re.sub(r"^```.*?\n", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```$", "", raw)

    try:
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Unable to deserialize obj: \n {raw} \nto JSON: {e}")

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise json.JSONDecodeError("No JSON found", raw, 0)


# always returns valid structured result
def _fallback(raw_output: str, chunks: List[Dict[str, Any]]) -> AnalysisResult:
    citations = []

    for chunk in chunks[:2]:
        citations.append(
            Citation(
                source=chunk.get("source", "unknown"),
                page_number=chunk.get("page"),
                excerpt=(chunk.get("text") or "")[:200],
            )
        )

    return AnalysisResult(
        answer=raw_output.strip() or "Unable to generate structured answer.",
        citations=citations,
        confidence=0.6,
    )


def _parse_response(raw_output: str, chunks: List[Dict[str, Any]]) -> AnalysisResult:
    try:
        data = _extract_json(raw_output)
        return AnalysisResult.model_validate(data)

    # fallback instead of failing
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"[Analyst] Failed structured parse: {e}")
        return _fallback(raw_output, chunks)


# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------


def analyst_node(state: ResearchState) -> Dict[str, Any]:
    """
    Synthesize retrieved chunks into a structured research response.

    - Build a prompt from the question, sub-task, and retrieved_chunks.
    - Invoke AWS Bedrock (e.g., Claude) with structured output enforcement.
    - Parse the response into an AnalysisResult.
    - Support streaming for real-time feedback.
    - Log actions to the scratchpad.

    Returns:
        Dict with "analysis" key containing the AnalysisResult as a dict,
        and "confidence_score" updated from the model's self-assessment.

    """
    question = state.get("user_question", "")
    task = state.get("current_task") or {}
    task_desc = task.get("description", "Analyze retrieved content")

    chunks = state.get("retrieved_chunks", [])

    context = _format_chunks(chunks)

    # FIX: pass correct variable
    prompt = _build_prompt(question, task_desc, context)

    logger.info(f"[Analyst] Starting analyst for: {question}")

    raw_output = _call_bedrock(prompt)
    result = _parse_response(raw_output, chunks)

    logger.info(f"[Analyst] Confidence: {result.confidence}")

    # advance plan
    plan_updates = _advance_plan(state, "analyst")
    logger.info("Advancing plan...")

    return {
        **plan_updates,
        "analysis_output": result.model_dump(),
        "confidence_score": result.confidence,
        "scratchpad": _append_scratchpad(
            state, f"Analyst generated response with confidence {result.confidence}"
        ),
    }

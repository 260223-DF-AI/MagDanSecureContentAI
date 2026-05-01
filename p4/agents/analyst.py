"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""
from typing import List, Optional, Dict, Any
import os
import json
import logging
from logs.log_config import setup_logging
from pydantic import BaseModel, ValidationError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
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
    """
    Convert retrieved chunks into a prompt-friendly string.
    """
    formatted = []
    for i, chunk in enumerate(chunks):
        formatted.append(
            f"[Chunk {i+1}] "
            f"Source: {chunk.get('source')} | "
            f"Page: {chunk.get('page', 'N/A')}\n"
            f"{chunk.get('text')}\n"
        )
    return "\n".join(formatted)


def _build_prompt(question: str, task: str, context: str) -> str:
    """
    Build the LLM prompt enforcing structured JSON output.
    """
    return f"""
You are an expert research analyst.

TASK:
{task}

QUESTION:
{question}

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question using ONLY the provided context.
- Provide citations (source + page_number if available).
- Include short supporting excerpts.
- Provide a confidence score between 0 and 1.

OUTPUT FORMAT (STRICT JSON):
{{
  "answer": "...",
  "citations": [
    {{
      "source": "...",
      "page_number": 1,
      "excerpt": "..."
    }}
  ],
  "confidence": 0.0
}}
"""


def _call_bedrock(prompt: str) -> str:
    """
    Invoke Bedrock Claude model.
    """
    llm = ChatBedrockConverse(
        model=os.getenv("BEDROCK_MODEL_ID"),
        temperature=0,
        region_name=os.getenv("AWS_REGION"),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def _stream_bedrock(prompt: str):
    """
    Streaming Bedrock response generator.
    """
    llm = ChatBedrockConverse(
        model=os.getenv("BEDROCK_MODEL_ID"),
        temperature=0,
        streaming=True,
        region_name=os.getenv("AWS_REGION"),
    )

    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


def _parse_response(raw_output: str) -> AnalysisResult:
    """
    Parse and validate LLM output.
    """
    try:
        data = json.loads(raw_output)
        return AnalysisResult(**data)
    except (json.JSONDecodeError, ValidationError):
        # Fallback if model fails formatting
        return AnalysisResult(
            answer=raw_output.strip(),
            citations=[],
            confidence=0.5,
        )

# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------

def analyst_node(state: ResearchState) -> Dict[str,Any]:
    """
    Synthesize retrieved chunks into a structured research response.

    TODO:
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
    task = state.get("current_task", {})
    task_desc = task.get("description")
    chunks = state.get("retrieved_chunks", [])

    context = _format_chunks(chunks)
    prompt = _build_prompt(question, task, context)

    streaming_enabled = os.getenv("ANALYST_STREAMING", "false").lower() == "true"

    logger.info(f"[Analyst] Starting analyst for: {question}")
    logger.debug(f"[Analyst] Current task: {task_desc}")

    if streaming_enabled:
        full_output = ""
        for token in _stream_bedrock(prompt):
            print(token, end="", flush=True)  # real-time feedback
            full_output += token
        raw_output = full_output
    else:
        raw_output = _call_bedrock(prompt)

    result = _parse_response(raw_output)

    # Advance plan
    logger.info("Advancing plan...")
    plan_updates = _advance_plan(state, "analyst")
    
    updates = _append_scratchpad(
            state,
            f"Analyst generated response with confidence {result.confidence}"
        )

    return {
        **plan_updates,
        "analysis_output": result.model_dump(),
        "confidence_score": result.confidence,
        "scratchpad": updates,
    }


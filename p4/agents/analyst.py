"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""
from typing import List, Optional, Dict, Any
import os
import json

from pydantic import BaseModel, ValidationError

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

from agents.state import ResearchState


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

def _append_scratchpad(state: ResearchState, message: str) -> List[str]:
    return state.get("scratchpad", []) + [message]


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
        model=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
        temperature=0,
        streaming=True,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
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
    task = state.get("current_task", {}).get("description", "")
    chunks = state.get("retrieved_chunks", [])

    context = _format_chunks(chunks)
    prompt = _build_prompt(question, task, context)

    streaming_enabled = os.getenv("ANALYST_STREAMING", "false").lower() == "true"

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
    plan = state.get("current_plan", [])
    index = state.get("current_task_index", 0)

    if index < len(plan):
        plan[index]["status"] = "complete"

    next_index = index + 1
    next_task = plan[next_index] if next_index < len(plan) else None

    return {
        "current_plan": plan,
        "current_task_index": next_index,
        "current_task": next_task,
        "analysis_output": result.dict(),
        "confidence_score": result.confidence,
        "scratchpad": _append_scratchpad(
            state,
            f"Analyst generated response with confidence {result.confidence}"
        ),
    }

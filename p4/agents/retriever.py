"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

import logging
import os

from infrastructure.instances import _get_embedder, _get_llm
from logs.log_config import setup_logging
from pinecone import Pinecone

from agents.state import ResearchState, _advance_plan, _append_scratchpad

setup_logging()
logger = logging.getLogger("researchflow.retriever")


def retriever_node(state: ResearchState) -> dict:
    """
    Retrieve relevant document chunks for the current sub-task.

    - Extract the current sub-task from state["plan"].
    - Query the Pinecone index with semantic search and metadata filters.
    - Apply context compression to reduce token noise.
    - Apply re-ranking to prioritize the most relevant results.
    - Return updated state with retrieved_chunks populated.
    - Log actions to the scratchpad.

    Returns:
        Dict with "retrieved_chunks" key containing a list of dicts,
        each with: content, relevance_score, source, page_number.

    """
    question = state["user_question"]
    task = state.get("current_task", {})
    task_desc = task.get("description", "retrieve relevant chunks")

    logger.info(f"[Retriever] Starting retrieval for: {question}")
    logger.debug(f"[Retriever] Current task: {task_desc}")

    # 1. Embed the query
    _embedder = _get_embedder
    query_vec = _embedder.embed_query(question)

    # 2. Query Pinecone
    pc = Pinecone(os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    # simple filters to ensure data is valid
    metadata_filter = {
        "source": {"$exists": True},
        "text": {"$exists": True},
        "page_number": {"$gte": 1},
    }

    res = index.query(
        vector=query_vec,
        top_k=12,
        include_metadata=True,
        namespace="primary-corpus",
        filter=metadata_filter,
    )

    raw_matches = res.get("matches", [])
    logger.info(f"[Retriever] Retrieved {len(raw_matches)} raw matches")

    # 3. Re-rank results
    docs_for_rerank = [{"text": m["metadata"]["text"]} for m in raw_matches]
    reranked = pc.inference.rerank(
        model="pinecone-rerank-v0",
        query=question,
        documents=docs_for_rerank,
        top_n=5,
        rank_fields=["text"],
    )

    # 4. Apply context compression
    retrieved_chunks = []
    for r in reranked.data:
        original_index = r.index
        match = raw_matches[original_index]
        md = match["metadata"]

        retrieved_chunks.append(
            {
                "chunk_id": match["id"],
                "content": _compress(md["text"], question),  # compression AFTER rerank
                "relevance_score": r.score,
                "source": md.get("source"),
                "page_number": md.get("page_number"),
            }
        )

    logger.info(
        f"[Retriever] Returning top {len(retrieved_chunks)} re-ranked and compressed chunks"
    )

    # 5. Log to scratchpad
    scratch_msg = (
        f"Retriever executed task: '{task_desc}'. "
        f"Retrieved {len(retrieved_chunks)} chunks for query: '{question}'."
    )

    updates = _append_scratchpad(state, scratch_msg)

    # 6. Advance plan and return
    logger.info("Advancing plan...")
    plan_updates = _advance_plan(state, "retriever")

    return {
        **plan_updates,
        "retrieved_chunks": retrieved_chunks,
        "scratchpad": updates,
    }


def _compress(text: str, question: str, max_tokens: int = 150) -> str:
    """
    Compress chunk text while preserving relevant context for the query.

    Invokes Bedrock chat model for compression.
    """
    prompt = f"""
    Extract only the sentences that directly help answer the question:
    '{question}'.

    Keep the extracted text under {max_tokens} tokens.
    Do not summarize — extract verbatim sentences only.
    Text:
    {text}
    """

    llm = _get_llm()

    return llm.invoke(prompt).content.strip()

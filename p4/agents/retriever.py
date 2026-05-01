"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""
import os
import logging
from pinecone import Pinecone
from logs.log_config import setup_logging
from langchain_aws import BedrockEmbeddings
from agents.state import ResearchState, _advance_plan, _append_scratchpad

setup_logging()
logger = logging.getLogger("researchflow.ingest")

def retriever_node(state: ResearchState) -> dict:
    """
    Retrieve relevant document chunks for the current sub-task.

    TODO:
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
    embedder = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    query_vec = embedder.embed_query(question)

    # 2. Query Pinecone
    pc = Pinecone(os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    # simple filters to ensure data is valid
    metadata_filter = {
        "source": {"$exists": True},
        "text": {"$exists": True},
        "page_number": {"$gte": 1}
    }   

    res = index.query(
        vector=query_vec,
        top_k=12,
        include_metadata=True,
        namespace="primary-corpus",
        filter=metadata_filter
    )

    raw_matches = res.get("matches", [])
    logger.info(f"[Retriever] Retrieved {len(raw_matches)} raw matches")
    
    # 3. Apply context compression
    compressed = []
    for m in raw_matches:
        md = m["metadata"]

        compressed.append({
            "chunk_id": md.get("id"),
            "content": compress(md.get("text", "")),
            "relevance_score": float(m.get("score", 0.0)),
            "source": md.get("source"),
        })
    
    # 4. Re-rank (highest to lowest score)
    reranked = sorted(compressed, key=lambda x: x["relevance_score"], reverse=True)
    retrieved_chunks = reranked[:5]
    logger.info(f"[Retriever] Returning top {len(retrieved_chunks)} re-ranked chunks")
    
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

def compress(text: str, max_len: int = 500) -> str:
    """
    Helper function to compress long chunks
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."

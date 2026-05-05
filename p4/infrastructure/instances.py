"""Creates instances for reused models and Pinecone indexes."""

import os

from langchain_aws import BedrockEmbeddings, ChatBedrock
from pinecone import Pinecone


def _get_embedder() -> BedrockEmbeddings:
    """
    Create a BedrockEmbeddings embedding model instance.

    Lazy-init so unit tests can monkeypatch before first call.
    """
    _embedder = None
    if _embedder is None:
        _embedder = BedrockEmbeddings(model=os.getenv("BEDROCK_EMBEDDING_MODEL_ID"))
    return _embedder


def _get_llm() -> ChatBedrock:
    """Create a ChatBedrock llm model instance."""
    _llm = None
    if _llm is None:
        _llm = ChatBedrock(
            model=os.getenv("BEDROCK_MODEL_ID"),
            temperature=0.1,
            region_name=os.getenv("AWS_REGION"),
        )
    return _llm


def _get_index() -> Pinecone.Index:
    """
    Fetch existing Pinecone index.

    Lazy-init so unit tests can monkeypatch before first call.
    """
    pc = Pinecone(os.getenv("PINECONE_API_KEY"))
    _pinceone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    return _pinceone_index

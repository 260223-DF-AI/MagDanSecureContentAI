"""
Unit Tests — Retriever Agent

Tests the retriever node using mocked Pinecone calls.
Validates re-ranking behavior and output structure.
"""

from unittest.mock import MagicMock


def _fake_match(content, score, source="doc.pdf", page=1):  # noqa: ANN001, ANN201, ANN202, D102
    return {
        "id": f"id-{content[:5]}",
        "score": score,
        "metadata": {"content": content, "source": source, "page_number": page},
    }


class TestRetrieverAgent:
    """Tests for retriever.py"""

    def _patched_index(self, matches):  # noqa: ANN001, ANN201, ANN202, D102
        index = MagicMock()
        index.query.return_value = {"matches": matches}
        return index

    def test_returns_structured_chunks(self, monkeypatch):  # noqa: ANN001, ANN201, D102
        from agents import retriever

        monkeypatch.setattr(
            retriever,
            "_get_index",
            lambda: self._patched_index(
                [
                    _fake_match("Apollo 11 landed on the Moon in 1969.", 0.91),
                    _fake_match("Apollo 13 had an oxygen tank failure.", 0.87),
                ]
            ),
        )
        out = retriever.retriever_node(
            {
                "user_question": "When did Apollo 11 land?",
                "current_plan": ["When did Apollo 11 land?"],
                "current_task_index": 0,
            }
        )
        assert "retrieved_chunks" in out
        assert len(out["retrieved_chunks"]) >= 1
        for c in out["retrieved_chunks"]:
            for k in ("content", "relevance_score", "source", "page_number"):
                assert k in c

    def test_applies_reranking(self, monkeypatch):  # noqa: ANN001, ANN201, D102
        from agents import retriever

        # Pinecone returns the WRONG order; rerank should reshuffle.
        monkeypatch.setattr(
            retriever,
            "_get_index",
            lambda: self._patched_index(
                [
                    _fake_match("Completely unrelated text about gardening.", 0.99),
                    _fake_match("Apollo 11 landed on the Moon in July 1969.", 0.10),
                ]
            ),
        )
        out = retriever.retriever_node(
            {
                "user_question": "When did Apollo 11 land?",
                "current_plan": {
                    "id": 1,
                    "task_type": "retrieve",
                    "description": "When did Apollo 11 land?",
                    "status": "pending",
                },
                "current_task_index": 0,
            }
        )
        # Top-1 after rerank should be the actually-relevant chunk.
        assert "Apollo" in out["retrieved_chunks"][0]["content"]

    def test_applies_context_compression(self, monkeypatch):  # noqa: ANN001, ANN201, D102
        from agents import retriever

        long = ". ".join([f"Sentence {i} about Apollo." for i in range(20)])
        monkeypatch.setattr(
            retriever,
            "_get_index",
            lambda: self._patched_index([_fake_match(long, 0.9)]),
        )
        out = retriever.retriever_node(
            {
                "user_question": "Apollo summary",
                "current_plan": {
                    "id": 1,
                    "task_type": "retrieve",
                    "description": "Apollo summary",
                    "status": "pending",
                },
                "current_task_index": 0,
            }
        )
        assert len(out["retrieved_chunks"][0]["content"]) < len(long)

    def test_handles_empty_results(self, monkeypatch):  # noqa: ANN001, ANN201, D102
        from agents import retriever

        monkeypatch.setattr(retriever, "_get_index", lambda: self._patched_index([]))
        out = retriever.retriever_node(
            {
                "user_question": "anything",
                "current_plan": {
                    "id": 1,
                    "task_type": "retrieve",
                    "description": "anything",
                    "status": "pending",
                },
                "current_task_index": 0,
            }
        )
        assert isinstance(out["retrieved_chunks"], list)

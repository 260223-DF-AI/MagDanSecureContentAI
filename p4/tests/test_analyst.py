"""
Unit Tests — Analyst Agent

Tests the analyst node using mocked Bedrock calls.
Validates structured output schema and confidence scoring.
"""

from unittest.mock import patch, MagicMock
import pytest
from agents.analyst import AnalysisResult, analyst_node
import json

class TestAnalystAgent:
    """Tests for agents.analyst.analyst_node."""

    def _sample_state(self):
        return {
            "user_question": "What is utilitarianism?",
            "current_task": {
                "id": 2,
                "task_type": "analyze",
                "description": "Synthesize an answer using retrieved chunks.",
                "status": "pending",
            },
            "current_plan": [
                {
                    "id": 1,
                    "task_type": "retrieve",
                    "description": "Retrieve relevant source chunks.",
                    "status": "complete",
                },
                {
                    "id": 2,
                    "task_type": "analyze",
                    "description": "Synthesize an answer using retrieved chunks.",
                    "status": "pending",
                },
                {
                    "id": 3,
                    "task_type": "fact_check",
                    "description": "Verify the answer.",
                    "status": "pending",
                },
            ],
            "current_task_index": 1,
            "retrieved_chunks": [
                {
                    "text": "Utilitarianism says actions are right if they promote the greatest happiness.",
                    "source": "utilitarianism_notes.txt",
                    "page": 1,
                    "chunk_id": "chunk_1",
                    "relevance_score": 0.92,
                }
            ],
            "analysis_output": {},
            "confidence_score": 0.0,
            "fact_check_results": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "scratchpad": [],
        }
    
    def _mock_response(self):
        response = MagicMock()
        response.content = json.dumps(
            {
                "answer": "Utilitarianism is an ethical theory focused on maximizing happiness.",
                "citations": [
                    {
                        "source": "utilitarianism_notes.txt",
                        "page_number": 1,
                        "excerpt": "actions are right if they promote the greatest happiness",
                    }
                ],
                "confidence": 0.92,
            }
        )
        return response

    @patch("agents.analyst.ChatBedrockConverse")
    def test_returns_valid_analysis_result(self, mock_chat_bedrock):
        """
        TODO:
        - Mock the Bedrock LLM invocation.
        - Call analyst_node with sample retrieved_chunks.
        - Assert the output parses into a valid AnalysisResult.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response()
        mock_chat_bedrock.return_value = mock_llm

        output = analyst_node(self._sample_state())
        result = AnalysisResult.model_validate(output["analysis_output"])

        assert isinstance(result, AnalysisResult)
        assert result.answer
        assert output["confidence_score"] == result.confidence

    @patch("agents.analyst.ChatBedrockConverse")
    def test_includes_citations(self, mock_chat_bedrock):
        """
        TODO:
        - Assert the AnalysisResult contains at least one Citation.
        - Assert citation source matches a retrieved chunk source.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response()
        mock_chat_bedrock.return_value = mock_llm

        output = analyst_node(self._sample_state())
        result = AnalysisResult.model_validate(output["analysis_output"])

        assert len(result.citations) >= 1
        assert result.citations[0].source == "utilitarianism_notes.txt"

    @patch("agents.analyst.ChatBedrockConverse")
    def test_confidence_within_range(self, mock_chat_bedrock):
        """
        TODO:
        - Assert confidence_score is between 0.0 and 1.0.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_response()
        mock_chat_bedrock.return_value = mock_llm

        output = analyst_node(self._sample_state())

        assert 0.0 <= output["confidence_score"] <= 1.0

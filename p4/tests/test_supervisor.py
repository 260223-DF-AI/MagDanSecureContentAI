"""
Unit Tests — Supervisor Graph

Tests the routing logic and conditional edges using mocked sub-agents.
"""

from unittest.mock import patch, MagicMock

import pytest

from agents.supervisor import planner_node, router, critique_node


class TestSupervisorRouting:
    """Tests for agents.supervisor routing and conditional edges."""

    @patch("agents.supervisor.ChatBedrock")
    def test_planner_decomposes_question(self, mock_chat_bedrock):
        """
        Mock the LLM call inside planner_node.
        Assert it populates state["plan"] with a non-empty list.
        """

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """
        1. Retrieve relevant documents about LangGraph.
        2. Analyze the retrieved context.
        3. Verify claims with fact-checking sources.
        """

        mock_llm.invoke.return_value = mock_response
        mock_chat_bedrock.return_value = mock_llm

        state = {
            "question": "How does LangGraph support multi-agent workflows?",
            "plan": [],
            "current_task": None,
            "retrieved_chunks": [],
            "analysis_output": None,
            "fact_check_results": [],
            "confidence_score": 0.0,
            "iteration_count": 0,
        }

        result = planner_node(state)

        assert "plan" in result
        assert isinstance(result["plan"], list)
        assert len(result["plan"]) > 0

    def test_router_selects_retriever(self):
        """
        Provide a state where the next sub-task requires retrieval.
        Assert router() returns "retriever".
        """

        state = {
            "plan": [
                {
                    "task": "Retrieve relevant context from Pinecone",
                    "type": "retrieval",
                }
            ],
            "current_task": {
                "task": "Retrieve relevant context from Pinecone",
                "type": "retrieval",
            },
            "retrieved_chunks": [],
            "analysis_output": None,
        }

        route = router(state)

        assert route == "retriever"

    def test_router_selects_analyst(self):
        """
        Provide a state where retrieval is complete.
        Assert router() returns "analyst".
        """

        state = {
            "plan": [
                {
                    "task": "Analyze retrieved chunks and synthesize answer",
                    "type": "analysis",
                }
            ],
            "current_task": {
                "task": "Analyze retrieved chunks and synthesize answer",
                "type": "analysis",
            },
            "retrieved_chunks": [
                {
                    "text": "LangGraph uses StateGraph to manage agent workflows.",
                    "source": "docs",
                }
            ],
            "analysis_output": None,
        }

        route = router(state)

        assert route == "analyst"

    def test_critique_triggers_retry(self):
        """
        Set confidence below threshold, iteration < max.
        Assert critique_node routes back for refinement.
        """

        state = {
            "confidence_score": 0.45,
            "iteration_count": 1,
            "max_iterations": 3,
            "analysis_output": "Partial answer with weak support.",
            "fact_check_results": [],
            "hitl_required": False,
        }

        result = critique_node(state)

        assert result["critique_decision"] == "retry"
        assert result["hitl_required"] is False
        assert result["iteration_count"] == 2

    def test_critique_triggers_hitl(self):
        """
        Set confidence below threshold, iteration >= max.
        Assert critique_node triggers HITL interrupt.
        """

        state = {
            "confidence_score": 0.45,
            "iteration_count": 3,
            "max_iterations": 3,
            "analysis_output": "Still low confidence after retries.",
            "fact_check_results": [],
            "hitl_required": False,
        }

        result = critique_node(state)

        assert result["critique_decision"] == "hitl"
        assert result["hitl_required"] is True

    def test_critique_accepts_response(self):
        """
        Set confidence above threshold.
        Assert critique_node routes to END.
        """

        state = {
            "confidence_score": 0.88,
            "iteration_count": 1,
            "max_iterations": 3,
            "analysis_output": "Final supported answer.",
            "fact_check_results": [
                {
                    "claim": "LangGraph supports conditional routing.",
                    "verified": True,
                }
            ],
            "hitl_required": False,
        }

        result = critique_node(state)

        assert result["critique_decision"] == "accept"
        assert result["hitl_required"] is False
        assert result["final_answer"] == "Final supported answer."
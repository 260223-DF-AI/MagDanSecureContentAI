"""
Unit Tests — Supervisor Graph

Tests the routing logic and conditional edges using mocked sub-agents.
"""

from unittest.mock import patch, MagicMock

import pytest


class TestSupervisorRouting:
    """Tests for agents.supervisor routing and conditional edges."""

    def test_planner_decomposes_question(self):
        """
        TODO:
        - Mock the LLM call inside planner_node.
        - Assert it populates state["plan"] with a non-empty list.
        """
        pass

    def test_router_selects_retriever(self):
        """
        TODO:
        - Provide a state where the next sub-task requires retrieval.
        - Assert router() returns "retriever".
        """
        pass

    def test_router_selects_analyst(self):
        """
        TODO:
        - Provide a state where retrieval is complete.
        - Assert router() returns "analyst".
        """
        pass

    def test_critique_triggers_retry(self):
        """
        TODO:
        - Set confidence below threshold, iteration < max.
        - Assert critique_node routes back for refinement.
        """
        pass

    def test_critique_triggers_hitl(self):
        """
        TODO:
        - Set confidence below threshold, iteration >= max.
        - Assert critique_node triggers HITL interrupt.
        """
        pass

    def test_critique_accepts_response(self):
        """
        TODO:
        - Set confidence above threshold.
        - Assert critique_node routes to END.
        """
        pass

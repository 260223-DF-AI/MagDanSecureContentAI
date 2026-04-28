"""
ResearchFlow — Input Guardrails Middleware

Detects and blocks prompt injection / stuffing attacks
in user inputs before they reach the agent pipeline.
"""


def detect_injection(user_input: str) -> bool:
    """
    Scan user input for common prompt injection patterns.

    TODO:
    - Check for system prompt override attempts.
    - Check for instruction stuffing patterns.
    - Return True if injection is detected, False otherwise.
    """
    raise NotImplementedError


def sanitize_input(user_input: str) -> str:
    """
    Clean user input by removing or escaping dangerous patterns.

    TODO:
    - Strip known injection markers.
    - Escape special formatting that could manipulate prompts.
    - Return the sanitized string.
    """
    raise NotImplementedError

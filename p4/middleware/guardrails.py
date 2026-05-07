"""
ResearchFlow — Input Guardrails Middleware

Detects and blocks prompt injection / stuffing attacks
in user inputs before they reach the agent pipeline.
"""

import re

# Patterns we treat as "almost certainly malicious".
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above|the\s+system)", re.I),
    re.compile(r"system\s*:\s*you\s+are", re.I),
    re.compile(r"</?\s*system\s*>", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
    re.compile(r"forget\s+everything", re.I),
    re.compile(r"reveal\s+your\s+(system\s+)?prompt", re.I),
]

# Markers we strip but don't reject for.
_STUFFING_MARKERS = [
    re.compile(r"```(?:system|assistant|developer)[\s\S]*?```", re.I),
    re.compile(r"<\s*/?(?:system|assistant)\s*>", re.I),
]

_MAX_INPUT_LEN = 4000  # characters


def detect_injection(user_input: str) -> bool:
    """
    Scan user input for common prompt injection patterns.

    - Check for system prompt override attempts.
    - Check for instruction stuffing patterns.
    - Return True if injection is detected, False otherwise.
    """
    if not user_input:
        return False
    if len(user_input) > _MAX_INPUT_LEN:
        return True
    return any(p.search(user_input) for p in _INJECTION_PATTERNS)


def sanitize_input(user_input: str) -> str:
    """
    Clean user input by removing or escaping dangerous patterns.

    - Strip known injection markers.
    - Escape special formatting that could manipulate prompts.
    - Return the sanitized string.
    """
    out = user_input or ""
    for marker in _STUFFING_MARKERS:
        out = marker.sub("", out)
    # Collapse repeated whitespace introduced by stripping.
    out = re.sub(r"\s{3,}", " ", out).strip()
    return out[:_MAX_INPUT_LEN]

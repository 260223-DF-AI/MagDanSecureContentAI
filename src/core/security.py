import re

# Basic input sanitization to prevent prompt injection attacks
# Required by project spec

def sanitize_user_text(text: str) -> str:
    """
    Removes dangerous prompt injection phrases.

    Example:
    "ignore previous instructions" → "[REDACTED]"
    """
    patterns = [
        r"ignore previous instructions",
        r"system prompt",
        r"developer message",
        r"reveal hidden instructions",
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "[REDACTED]", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()
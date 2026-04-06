import re


def sanitize_user_text(text: str) -> str:
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
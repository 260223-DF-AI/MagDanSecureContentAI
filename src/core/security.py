import re


def sanitize_user_text(text: str) -> str:
    patterns = [
        r"ignore previous instructions",
        r"system prompt",
        r"developer message",
    ]

    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "[REDACTED]", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()
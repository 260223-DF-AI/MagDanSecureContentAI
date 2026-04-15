import re
import unicodedata

DEFAULT_POLICY = """
You are an expert content moderation agent for the HerSpace social media platform.

Your task:
- Analyze text descriptions and determine if the content aligns with HerSpace Policies listed below.
- Return your final answer (moderation decision) as: "safe" or "unsafe".
"unsafe" marking content that does not adhere to policies.

Policies:
1. Topics of violence, self-harm, or harassment are prohibited.
2. Post descriptions should pass the Bechdel test meaning the central purpose of the text should not be centered around a man.
    Example violation: "I love my man. He's the best husband ever. He's my whole world."
    Violation reasoning: "This description is in violation of the Bechdel test policy on HerSpace. Content on this platform should not be centered around men"
3. The following phrases are prohibited from post descriptions:
    "guys night out", "boys night out", "out with the boys", "man cave", "my boysss", "alpha male", "sigma male"
4. These phrases are strictly allowed as they promote HerSpace mission statement: "girls night out", "girls day"
5. Profanity is strictly prohibited.
    
Example content moderation:
    Description 1: "Happy at work! My coworker, Charles, helped me out today.
    Decision 1: "safe"
    
    Description 2: "Bachelor weekend!! "Night out with my guyss"
    Decision 2: "unsafe"
    
    Description 3: "girls night out!!"
    Decision 3: "safe"

{
  "reasoning": "short explanation",
  "decision": "safe" or "unsafe"
}
"""

# Prompt injection protection
MAX_INPUT_CHARS = 4000
INJECTION_PATTERNS = [
    r"ignore (all |your )?(previous |prior )?instructions",
    r"you are now",
    r"disregard (all |your )?(previous |prior )?(instructions|context)",
    "ignore previous instructions",
    "disregard above",
    "you are no longer",
    "system:",
    "<system>",
    "</system>",
    "assistant:",
    "###",
]

def validate_and_sanitize(user_input: str) -> str:
    """Validate and sanitize user input before sending to an LLM."""
    # 1. Enforce length limit
    if len(user_input) > MAX_INPUT_CHARS:
        raise ValueError(f"Input exceeds max length of {MAX_INPUT_CHARS} chars.")

    # 2. Strip leading/trailing whitespace
    sanitized = user_input.strip()
    
    # 3. Reject empty input
    if not sanitized:
        raise ValueError("Input cannot be empty.")
    
    # 4. Normalize Unicode (prevents homoglyph attacks)
    sanitized = unicodedata.normalize("NFKC", sanitized)
    
    # 5. Remove null bytes and control characters
    sanitized = re.sub(r"[\x00-\x1F\x7F]", " ", sanitized)
    
    # 6. Collapse repeated whitespace
    sanitized = re.sub(r"\s+", " ", sanitized)

    # 7. Detect injection patterns (case-insensitive)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            raise ValueError("Potentially malicious input detected. Request rejected.")

    return sanitized

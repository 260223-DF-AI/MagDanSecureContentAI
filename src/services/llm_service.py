from typing import Any


class LLMService:
    """
    Temporary text moderation / reasoning layer.

    Replace the keyword rules with:
    - OpenAI API
    - AWS Bedrock
    - SageMaker-hosted LLM
    - or your team's final prompt pipeline
    """

    def __init__(self) -> None:
        self.flagged_terms = {
            "hate",
            "threat",
            "kill",
            "violence",
            "slur",
        }

    def analyze_text(self, text: str, vision_result: dict[str, Any]) -> dict[str, Any]:
        lowered = text.lower()
        matched = [term for term in self.flagged_terms if term in lowered]

        if matched:
            moderation_label = "flagged"
            is_comment_allowed = False
            reason = (
                f"LLM moderation flagged the text because it matched restricted terms: {matched}."
            )
            suggested_response = "Policy Violation: This content cannot be posted."
        else:
            moderation_label = "safe"
            is_comment_allowed = True
            reason = "LLM moderation found no disallowed language patterns in the submitted text."
            suggested_response = "Approved"

        # Optional cross-signal example
        if not vision_result["is_post_allowed"]:
            moderation_label = "flagged"
            is_comment_allowed = False
            reason = (
                reason
                + " The associated image was also blocked by the vision model."
            )
            suggested_response = "Policy Violation: Image or text violates moderation policy."

        return {
            "moderation_label": moderation_label,
            "is_comment_allowed": is_comment_allowed,
            "reason": reason,
            "suggested_response": suggested_response,
        }
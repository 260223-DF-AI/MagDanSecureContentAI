from src.schemas import ReasoningResult, VisionResult
from src.services.interfaces import ReasoningEngine


class StubReasoningEngine(ReasoningEngine):
    async def analyze_text_and_vision(
        self,
        user_text: str,
        vision_result: VisionResult,
    ) -> ReasoningResult:

        if vision_result.label == "Flagged (Unsafe)":
            return ReasoningResult(
                decision="policy_violation",
                generated_response="Post violates policy.",
                trace={"reason": "unsafe image"},
            )

        if "hate" in user_text.lower():
            return ReasoningResult(
                decision="policy_violation",
                generated_response="Post contains unsafe language.",
                trace={"reason": "unsafe text"},
            )

        return ReasoningResult(
            decision="approved",
            generated_response=None,
            trace={"reason": "safe"},
        )
from src.schemas import AnalyzeResponse
from src.services.interfaces import VisionClassifier, ReasoningEngine


class ModerationPipeline:
    def __init__(
        self,
        vision_classifier: VisionClassifier,
        reasoning_engine: ReasoningEngine,
    ):
        self.vision_classifier = vision_classifier
        self.reasoning_engine = reasoning_engine

    async def run(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        user_text: str,
    ) -> AnalyzeResponse:

        vision_result = await self.vision_classifier.classify_image(
            file_bytes=file_bytes,
            filename=filename,
        )

        reasoning_result = await self.reasoning_engine.analyze_text_and_vision(
            user_text=user_text,
            vision_result=vision_result,
        )

        return AnalyzeResponse(
            image_label=vision_result.label,
            image_confidence=vision_result.confidence,
            final_decision=reasoning_result.decision,
            generated_response=reasoning_result.generated_response,
            trace=reasoning_result.trace,
        )
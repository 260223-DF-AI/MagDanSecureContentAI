from abc import ABC, abstractmethod
from src.schemas import VisionResult, ReasoningResult


class VisionClassifier(ABC):
    @abstractmethod
    async def classify_image(self, file_bytes: bytes, filename: str) -> VisionResult:
        raise NotImplementedError


class ReasoningEngine(ABC):
    @abstractmethod
    async def analyze_text_and_vision(
        self,
        user_text: str,
        vision_result: VisionResult,
    ) -> ReasoningResult:
        raise NotImplementedError
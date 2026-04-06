from abc import ABC, abstractmethod
from src.schemas import CNNTrainingSchema, LLMTrainingSchema


class CNNClassifier(ABC):
    @abstractmethod
    async def classify_image(
        self,
        file_bytes: bytes,
        filename: str,
        image_id: str,
    ) -> CNNTrainingSchema:
        raise NotImplementedError


class LLMReasoningEngine(ABC):
    @abstractmethod
    async def analyze_description(
        self,
        text: str,
        description_id: str,
        classification_cat: str,
    ) -> tuple[LLMTrainingSchema, dict]:
        raise NotImplementedError
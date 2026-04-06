from abc import ABC, abstractmethod
from src.schemas import CNNTrainingSchema, LLMTrainingSchema

# =========================
# ABSTRACT INTERFACES
# =========================
# These define REQUIRED behavior
# but do NOT implement it yet.
#
# Later:
# - Replace with SageMaker endpoints
# - Keep same interface → no API changes needed
# =========================

class CNNClassifier(ABC):
    @abstractmethod
    async def classify_image(
        self,
        file_bytes: bytes,
        filename: str,
        image_id: str,
    ) -> CNNTrainingSchema:
        """
        Takes an image and returns classification results.

        Future:
        - Call SageMaker endpoint
        - Return prediction + confidence
        """
        raise NotImplementedError


class LLMReasoningEngine(ABC):
    @abstractmethod
    async def analyze_description(
        self,
        text: str,
        description_id: str,
        classification_cat: str,
    ) -> tuple[LLMTrainingSchema, dict]:
        """
        Takes text + CNN output and performs reasoning.

        Uses:
        - ReAct prompting (required by project)

        Returns:
        - LLMTrainingSchema (model output)
        - trace (reasoning steps)
        """
        raise NotImplementedError
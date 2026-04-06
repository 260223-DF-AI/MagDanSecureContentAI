from src.schemas import VisionResult
from src.services.interfaces import VisionClassifier


class StubVisionClassifier(VisionClassifier):
    async def classify_image(self, file_bytes: bytes, filename: str) -> VisionResult:
        # Temporary logic — replace with SageMaker later
        if "unsafe" in filename.lower():
            return VisionResult(label="Flagged (Unsafe)", confidence=0.95)
        return VisionResult(label="Social", confidence=0.80)
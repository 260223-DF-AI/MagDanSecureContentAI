from src.schemas import CNNTrainingSchema
from src.services.interfaces import CNNClassifier


class StubCNNClassifier(CNNClassifier):
    async def classify_image(
        self,
        file_bytes: bytes,
        filename: str,
        image_id: str,
    ) -> CNNTrainingSchema:
        lower_name = filename.lower()

        if "unsafe" in lower_name or "flag" in lower_name:
            label = "Flagged (Unsafe)"
            confidence = 0.95
        elif "pro" in lower_name:
            label = "Professional"
            confidence = 0.88
        else:
            label = "Social"
            confidence = 0.81

        return CNNTrainingSchema(
            cnn_train_id=f"cnn-{image_id}",
            confidence_score=confidence,
            classification_cat=label,
            is_correct=None,
            accuracy=None,
            image_key=image_id,
        )
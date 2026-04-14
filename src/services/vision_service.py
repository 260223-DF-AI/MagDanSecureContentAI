import io
from typing import Any

from PIL import Image


class VisionService:
    """
    Temporary vision layer for SecureContent AI.

    Replace the mock logic in analyze_image() with your real model loading
    and inference call when your SageMaker/local model is ready.
    """

    def __init__(self) -> None:
        self.class_names = ["groups", "men", "women"]

    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        """
        Current behavior:
        - validates image is readable
        - returns mock/demo classification logic

        Later behavior:
        - preprocess image
        - run ResNet18 or SageMaker endpoint inference
        - map prediction to moderation decision
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size

        # Demo-only placeholder logic:
        # Replace with your trained model prediction.
        if "men" in filename.lower():
            predicted_class = "men"
            confidence = 0.93
            class_probabilities = {
                "groups": 0.03,
                "men": 0.93,
                "women": 0.04,
            }
        elif "group" in filename.lower():
            predicted_class = "groups"
            confidence = 0.88
            class_probabilities = {
                "groups": 0.88,
                "men": 0.05,
                "women": 0.07,
            }
        else:
            predicted_class = "women"
            confidence = 0.91
            class_probabilities = {
                "groups": 0.04,
                "men": 0.02,
                "women": 0.94,
            }

        is_post_allowed = predicted_class != "men"

        reason = (
            f"Vision model classified the image as '{predicted_class}' "
            f"with confidence {confidence:.2f}. "
            f"Image size detected: {width}x{height}. "
            f"{'Blocked because image was classified as men.' if not is_post_allowed else 'Image passed visual moderation.'}"
        )

        return {
            "predicted_class": predicted_class,
            "confidence_score": confidence,
            "class_probabilities": class_probabilities,
            "is_post_allowed": is_post_allowed,
            "reason": reason,
        }
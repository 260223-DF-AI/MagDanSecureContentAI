import io
from typing import Any

import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.core.settings import settings

from dotenv import load_dotenv
load_dotenv()

# invokes SageMaker endpoint directly for the vision/image classifier
class VisionService:
    """
    Vision layer backed by a deployed SageMaker endpoint.

    Expected endpoint behavior:
    - accepts raw image bytes
    - supports content types like image/jpeg, image/png, image/webp
    - returns JSON shaped like your inference.py output:
      {
        "predicted_class": "...",
        "confidence_score": 0.99,
        "class_probabilities": {...},
        "is_post_allowed": true
      }
    """

    def __init__(self) -> None:
        if not settings.vision_endpoint_name:
            raise ValueError(
                "VISION_ENDPOINT_NAME is not set. "
                "Set it in your environment before starting FastAPI."
            )

        # use SageMaker Runtime client for endpoint inference
        self.runtime = boto3.client(
            "sagemaker-runtime",
            region_name=settings.aws_region,
        )
        self.endpoint_name = settings.vision_endpoint_name

    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> dict[str, Any]:
        try:
            # CHANGE: send raw image bytes directly to SageMaker endpoint
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType=content_type,
                Accept="application/json",
                Body=image_bytes,
            )

            raw_body = response["Body"].read().decode("utf-8")
            prediction = json.loads(raw_body)

            # CHANGE: normalize result shape and add API-friendly explanation
            predicted_class = prediction.get("predicted_class", "unknown")
            confidence_score = float(prediction.get("confidence_score", 0.0))
            class_probabilities = prediction.get("class_probabilities", {})
            is_post_allowed = bool(prediction.get("is_post_allowed", False))

            reason = (
                f"SageMaker vision endpoint '{self.endpoint_name}' classified "
                f"'{filename}' as '{predicted_class}' with confidence "
                f"{confidence_score:.4f}."
            )

            return {
                "predicted_class": predicted_class,
                "confidence_score": confidence_score,
                "class_probabilities": class_probabilities,
                "is_post_allowed": is_post_allowed,
                "reason": reason,
            }

        except (ClientError, BotoCoreError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"SageMaker vision inference failed: {exc}") from exc
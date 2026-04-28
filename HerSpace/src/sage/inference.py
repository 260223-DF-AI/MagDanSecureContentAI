# in friday recording - 1:06

import io
import json
import os
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Keep these aligned with the actual class-folder names used in training.
# If your ImageFolder class order differs, update this list to match the
# classes saved in model_metadata.json.
DEFAULT_CLASS_NAMES = ["groups", "men", "women"]


class HumanIdentificationModel(nn.Module):
    """
    ResNet18-based classifier matching train.py.
    """
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _build_inference_transform() -> transforms.Compose:
    """
    Must match the evaluation transform used during training.
    """
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def model_fn(model_dir: str) -> dict[str, Any]:
    """
    Load model artifacts from /opt/ml/model.

    Expected files:
    - model.pth
    - model_metadata.json (optional but recommended)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    model_path = os.path.join(model_dir, "model.pth")

    class_names = DEFAULT_CLASS_NAMES
    num_classes = len(class_names)

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        saved_classes = metadata.get("classes")
        if isinstance(saved_classes, list) and saved_classes:
            class_names = saved_classes
            num_classes = len(class_names)

    model = HumanIdentificationModel(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "device": device,
        "transform": _build_inference_transform(),
        "class_names": class_names,
    }


def input_fn(request_body: bytes, request_content_type: str) -> torch.Tensor:
    """
    Convert incoming image bytes into a model-ready tensor.

    Supported content types:
    - image/jpeg
    - image/png
    - image/webp
    """
    supported_types = {"image/jpeg", "image/png", "image/webp"}

    if request_content_type not in supported_types:
        raise ValueError(
            f"Unsupported content type: {request_content_type}. "
            f"Supported types: {sorted(supported_types)}"
        )

    image = Image.open(io.BytesIO(request_body)).convert("RGB")
    tensor = _build_inference_transform()(image).unsqueeze(0)
    return tensor


def predict_fn(input_data: torch.Tensor, model_artifacts: dict[str, Any]) -> dict[str, Any]:
    """
    Run model inference and return structured prediction output.
    """
    model = model_artifacts["model"]
    device = model_artifacts["device"]
    class_names = model_artifacts["class_names"]

    input_data = input_data.to(device)

    with torch.no_grad():
        logits = model(input_data)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_index = int(predicted_idx.item())
    predicted_label = class_names[predicted_index]
    predicted_confidence = float(confidence.item())

    all_scores = {
        class_names[i]: float(probabilities[0, i].item())
        for i in range(len(class_names))
    }

    # Optional business rule mapping for your project demo.
    # Adjust if your moderation policy changes.
    is_allowed = predicted_label != "men"

    return {
        "predicted_class": predicted_label,
        "confidence_score": predicted_confidence,
        "class_probabilities": all_scores,
        "is_post_allowed": is_allowed,
    }


def output_fn(prediction: dict[str, Any], accept: str) -> tuple[str, str]:
    """
    Serialize prediction to JSON.
    """
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")

    return json.dumps(prediction), "application/json"
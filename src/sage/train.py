# done in Script mode in SageMaker Studio - matches the structure as classifyer
# just showing in the repo, so it's there to go back to

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.cuda.amp import GradScaler, autocast


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------
# Data
# -------------------------
def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transforms() -> transforms.Compose:
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


# -------------------------
# Model
# -------------------------
class HumanIdentificationModel(nn.Module):
    """
    ResNet18-based classifier for:
    - Professional
    - Social
    - Flagged (Unsafe)

    Adjust class name mapping to match your dataset folder names later if needed.
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -------------------------
# Train / Eval helpers
# -------------------------
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += inputs.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def train(
    args: argparse.Namespace,
) -> None:
    logger.info("Starting SageMaker training job")
    logger.info("Arguments: %s", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and args.use_amp

    train_dir = Path(args.train)
    test_dir = Path(args.test)
    model_dir = Path(args.model_dir)
    output_data_dir = Path(args.output_data_dir)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=build_transforms(),
    )
    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=build_eval_transforms(),
    )

    num_classes = len(train_dataset.classes)
    logger.info("Detected classes: %s", train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = HumanIdentificationModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda',enabled=use_amp)

    best_test_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += (preds == labels).sum().item()
            running_count += inputs.size(0)

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        logger.info(
            "Epoch %s/%s | train_loss=%.4f train_acc=%.4f | test_loss=%.4f test_acc=%.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0

            model_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), model_dir / "model.pth")

            metadata = {
                "classes": train_dataset.classes,
                "num_classes": num_classes,
                "best_test_loss": best_test_loss,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "image_size": [256, 256],
            }

            with open(model_dir / "model_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info("Saved best model to %s", model_dir)

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(
                    "Early stopping triggered after %s epochs without improvement",
                    args.patience,
                )
                break

    output_data_dir.mkdir(parents=True, exist_ok=True)
    with open(output_data_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_test_loss": best_test_loss,
                "epochs_requested": args.epochs,
                "patience": args.patience,
                "use_amp": use_amp,
            },
            f,
            indent=2,
        )

    logger.info("Training complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_amp", type=lambda x: str(x).lower() == "true", default=True)

    # SageMaker environment paths
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/Train"),
    )
    parser.add_argument(
        "--test",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/Test"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
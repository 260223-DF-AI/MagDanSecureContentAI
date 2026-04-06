import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy

DATA_ROOT = "data/humans"
TRAIN_DIR = os.path.join(DATA_ROOT, "training")
TEST_DIR = os.path.join(DATA_ROOT, "test")

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset directory '{TRAIN_DIR}' not found.")
    print("Please ensure the 'data/humans' folder is extracted in the script's directory.")
    sys.exit(1)

if not os.path.exists(TEST_DIR):
    print(f"ERROR: Dataset directory '{TEST_DIR}' not found.")
    print("Please ensure the 'data/humans' folder is extracted in the script's directory.")
    sys.exit(1)

data_transforms = transforms.Compose([
    transforms.Resize(tuple((256, 256))),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]  # filepath
        return img, label, path

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=data_transforms)
test_dataset = ImageFolderWithPaths(root=TEST_DIR, transform=data_transforms)

print(f"Classes found: {train_dataset.classes}")
print(f"Total training images available: {len(train_dataset)}")

# saved for db dimension later on
# idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
# idx_to_class[1]   # → "men"
# idx_to_class[2]   # → "groups"

BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# TODO: add comments
# send confidence score to DB model

class HumanIdentificationModel(nn.Module):
    """Model to identify images of humans and detect whether the images is of an individual man or woman or a group of people

    Args:
        nn.Module (class): super class for neural network modules.
    """
    def __init__(self):
        super(HumanIdentificationModel, self).__init__()
        self.flatten = nn.Flatten()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 256x256 -> 128x128

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 128x128 -> 64x64

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x64 -> 32x32
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(65536, 3)
        )

    def forward(self, x, probability_flag=False):
        x = self.classify(self.features(x))
        
        if probability_flag:
            return F.softmax(x, dim=1) # gets confidence scores in multi-class models
        return x

class LitClassifier(pl.LightningModule):
    def __init__(self, model, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task='multiclass', num_classes=3)
        
        # buffer for storing predictions each epoch
        self.validation_outputs = []
    
    def forward(self, x, probability_flag=False):
        return self.model(x, probability_flag=probability_flag)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Runs once per batch.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
        """
        X, y, paths = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        
        # accuracy
        prob = self(X, probability_flag=True)
        confidence, predicted_class = torch.max(prob, dim=1)
        acc = self.val_accuracy(predicted_class, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        
        # store for epoch-end processing
        self.validation_outputs.append({
            "pred": predicted_class.cpu(),
            "conf": confidence.cpu(),
            "y": y.cpu(),
            "paths": paths
        })
    
    def on_validation_epoch_end(self):
        """Runs every epoch.

        Args:
            outputs (_type_): _description_
        """
        for batch in self.validation_outputs:
            preds = batch["pred"]
            confs = batch["conf"]
            ys = batch["y"]
            paths = batch["paths"]
            
            for i in range(len(ys)):
                # Add DB logic
                if preds[i] != ys[i]:
                    print(
                    f"File: {paths[i]} | "
                    f"Class: {ys[i].item()} | "
                    f"Pred: {preds[i].item()} | "
                    f"Confidence: {confs[i].item():.4f}")
            
            
        # clear buffer for next epoch
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0001)

def train_model(train_loader, test_loader):
    model = HumanIdentificationModel()    
    lit_model = LitClassifier(model)
    print(lit_model)
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        mode='min'
    )
    
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    
    # NOTE: CPU-only machines (no NVIDIA GPU), cannot use AMP.
    # To use mixed precision (AMP), change precision=32 → precision=16
    # AND set accelerator="gpu" on a CUDA-capable system.
    trainer = pl.Trainer(
        max_epochs=2,
        precision=32, # switch to 16 to enable AMP on GPU
        accelerator='auto',
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=1,
        limit_train_batches=200,
        num_sanity_val_steps=0
    )
    
    trainer.fit(lit_model, train_loader, test_loader)
    return lit_model

def main():
    train_model(train_loader, test_loader)
    print("\nTraining complete! Run 'tensorboard --logdir lightning_logs/' to view.")

if __name__ == "__main__":
    main()

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torchvision import datasets, transforms, models
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sqlalchemy.orm import Session
from utils.audit_db.add_img_dataset_to_db import check_img_exists_in_db
from src.models.orm_models import CNNTraining, CNNTrainingRun
from src.models.instances import get_engine
from src.models.schemas import CNNTrainingSchema

DATA_ROOT = 'utils/data/humans'
TRAIN_DIR = os.path.join(DATA_ROOT, "training")
TEST_DIR = os.path.join(DATA_ROOT, "test")
engine = get_engine() # used for DB logging

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset directory '{TRAIN_DIR}' not found.") # TODO: change to logger
    print("Please ensure the 'utils/data/humans' folder is extracted in the script's directory.")
    sys.exit(1)

if not os.path.exists(TEST_DIR):
    print(f"ERROR: Dataset directory '{TEST_DIR}' not found.")
    print("Please ensure the 'utils/data/humans' folder is extracted in the script's directory.")
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
        path = self.samples[index][0]  # image filepath
        return img, label, path

train_dataset = ImageFolderWithPaths(root=TRAIN_DIR, transform=data_transforms)
test_dataset = ImageFolderWithPaths(root=TEST_DIR, transform=data_transforms)

print(f"Classes found: {train_dataset.classes}") # TODO: change to use logger
print(f"Total training images available: {len(train_dataset)}")

BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

class HumanIdentificationModel(nn.Module):
    """Model to identify images of humans and detect whether the images is of an individual man or woman or a group of people
    Using ResNet18-based model for human classification.
    Args:
        nn.Module (class): super class for neural network modules.
    """
    def __init__(self):
        super(HumanIdentificationModel, self).__init__()
        # load pretrained model - ResNet18
        self.model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

        # freeze early layers
        for param in self.model.parameters():
            param.requires_grad = False

        # replace final layer (resnet default is 1000 classes) to 3
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 3)

    def forward(self, x, probability_flag=False):
        x = self.model(x)
        
        if probability_flag:
            return torch.softmax(x, dim=1) # gets confidence scores in multi-class models
        return x

class LitClassifier(pl.LightningModule):
    def __init__(self, model, run_id=None, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task='multiclass', num_classes=3)
        self.run_id = run_id
        
        # buffers for storing predictions each epoch
        self.training_outputs = []
        self.validation_outputs = []
    
    def forward(self, x, probability_flag=False):
        return self.model(x, probability_flag=probability_flag)
    
    def training_step(self, batch, batch_idx):
        X, y, paths = batch
        pred = self(X)
        loss = self.loss_fn(pred, y)
        
        # accuracy - gets confidence score for audit logs
        prob = self(X, probability_flag=True)
        confidence, predicted_class = torch.max(prob, dim=1)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        
        self.training_outputs.append({
            "pred": predicted_class.cpu(),
            "conf": confidence.cpu(),
            "y": y.cpu(),
            "paths": paths
        })
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Runs once per batch."""
        
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
        """Runs every epoch."""
        
        self._send_training_data_to_db()
            
        # clear buffers for next epoch
        self.validation_outputs.clear()
        self.training_outputs.clear()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0001)

    def _send_training_data_to_db(self):
        """Method to send training data to cnn_training table of DB. Runs on evaluation data at the end of every epoch.
        
        Data gathered:
            - Image file path: str
            - Image id (FK)
            - Run id (FK)
            - Correct label: str
            - Predicted label: str
            - AI confidence score: float
            - is_correct: bool
        """
        
        with Session(engine) as session:
            try:
                for v_batch in self.validation_outputs:
                    self.__data_to_db_inner_loop(v_batch, session)
                
                for t_batch in self.training_outputs:
                    self.__data_to_db_inner_loop(t_batch, session)
                    
                session.commit() # Commit all CNN rows per at once
                
            except Exception as e:
                session.rollback()
                raise ConnectionError(f"DB commit failed: {e}")
                
        print("Successfully sent training data to db!\n") # TODO: switch to logger

    def __data_to_db_inner_loop(self, batch, session):
        """Private helper method to add entries to cnn_training table

        Args:
            batch (dict): dictionary holding predictions, confidence scores, labels, and image paths
            session (Session): orm session for DML
        """
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()} # dictionary of class names and idxs

        preds = batch["pred"]
        confs = batch["conf"]
        ys = batch["y"]
        paths = batch["paths"]
        
        for i in range(len(ys)):
            file_path = paths[i]
            true_name = idx_to_class[int(ys[i].item())]
            pred_name = idx_to_class[int(preds[i].item())]
            confidence = float(confs[i].item())
            is_correct = true_name == pred_name

            # 1. Verify  image exists in dim_image table
            img_row = check_img_exists_in_db(file_path, true_name, session) # adds new entry, in none exists

            # 2. Insert CNNTraining row
            cnn_schema = CNNTrainingSchema(
                confidence_score=confidence,
                predicted_class=str(pred_name),
                is_correct=is_correct,
                image_key=img_row.image_id,
                run_key=self.run_id
            ) # create object

            cnn_row = CNNTraining(**cnn_schema.model_dump()) # convert obj to ORM

            session.add(cnn_row)

def train_model(train_loader, test_loader, run_id: int):
    model = HumanIdentificationModel()    
    lit_model = LitClassifier(model, run_id)
    print(lit_model)
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=1,
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
        max_epochs=100,
        precision=32, # switch to 16 to enable AMP on GPU
        accelerator='auto',
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=1,
        limit_train_batches=400,
        num_sanity_val_steps=0
    )
    
    trainer.fit(lit_model, train_loader, test_loader)
    return lit_model

def main():
    # log training session to db
    # adds new row to cnn_training_runs table
    with Session(engine) as session:
        run = CNNTrainingRun()
        session.add(run)
        session.commit()
        session.refresh(run)
        training_run_id = run.run_id # used as FK in cnn_training table
    
    train_model(train_loader, test_loader, training_run_id)
    print("\nTraining complete! Run 'tensorboard --logdir lightning_logs/' to view.")

if __name__ == "__main__":
    main()

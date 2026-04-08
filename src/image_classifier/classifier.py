import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy
from sqlalchemy.orm import Session 
from src.models.orm_models import CNNTraining, CNNTrainingRun, DimImage
from src.models.instances import get_engine
from src.models.schemas import CNNTrainingSchema, DimImageSchema

DATA_ROOT = 'utils/data/humans'
TRAIN_DIR = os.path.join(DATA_ROOT, "training")
TEST_DIR = os.path.join(DATA_ROOT, "test")
engine = get_engine()

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset directory '{TRAIN_DIR}' not found.")
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

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=data_transforms)
test_dataset = ImageFolderWithPaths(root=TEST_DIR, transform=data_transforms)

print(f"Classes found: {train_dataset.classes}") # TODO: change to use logger
print(f"Total training images available: {len(train_dataset)}")

BATCH_SIZE = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# TODO: add comments
# send confidence score to DB model

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
            return F.softmax(x, dim=1) # gets confidence scores in multi-class models
        return x

class LitClassifier(pl.LightningModule):
    def __init__(self, model, run_id=None, imgs=None, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task='multiclass', num_classes=3)
        self.run_id = run_id
        self.existing_imgs = imgs
        
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
        try:
            self._send_training_data_to_db()
        except:
            raise ConnectionError('Could not upload data to DB.')
            
        # clear buffer for next epoch
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0001)

    def _send_training_data_to_db(self):
        # add paths to image table - not in this function - DB setup
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        
        with Session(engine) as session:
            for batch in self.validation_outputs:
                preds = batch["pred"]
                confs = batch["conf"]
                ys = batch["y"]
                paths = batch["paths"]
                
                for i in range(len(ys)):
                    if preds[i] != ys[i]:
                        file_path = paths[i]
                        true_name = idx_to_class[int(ys[i].item())]
                        pred_name = idx_to_class[int(preds[i].item())]
                        confidence = float(confs[i].item())
                        is_correct = true_name == pred_name

                        # 1. Ensure image exists in dim_image table
                        if file_path not in self.existing_imgs:
                            # Create Pydantic schema
                            img_schema = DimImageSchema(
                                image_path=file_path,
                                correct_cat=str(true_name)
                            )

                            # Convert to ORM
                            img_row = DimImage(**img_schema.model_dump())
                            session.add(img_row)
                            session.commit()
                            session.refresh(img_row)

                            # Add to in-memory set
                            self.existing_imgs.add(file_path)

                        else:
                            # Fetch existing image row
                            img_row = session.query(DimImage).filter_by(image_path=file_path).first()

                        # 2. Insert CNNTraining row
                        cnn_schema = CNNTrainingSchema(
                            confidence_score=confidence,
                            predicted_class=str(pred_name),
                            is_correct=is_correct,
                            image_key=img_row.image_id,
                            run_key=self.run_id
                        )

                        cnn_row = CNNTraining(**cnn_schema.model_dump())

                        session.add(cnn_row)

                # Commit all CNN rows at once
                session.commit()
                print("Successfully sent training data to db!") # TODO: switch to logger

def preload_existing_images(engine):
    """Load all image paths from the DB into a fast lookup set."""
    with Session(engine) as session:
        rows = session.query(DimImage.image_path).all()
        return {row[0] for row in rows}
    
def train_model(train_loader, test_loader, run_id: int, existing_imgs: set):
    model = HumanIdentificationModel()    
    lit_model = LitClassifier(model, run_id, existing_imgs)
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
    existing_imgs = preload_existing_images(engine)

    # log training session to db
    # adds new row to cnn_training_runs table
    with Session(engine) as session:
        run = CNNTrainingRun()
        session.add(run)
        session.commit()
        session.refresh(run)
        training_run_id = run.run_id # used as FK in cnn_training table
    
    train_model(train_loader, test_loader, training_run_id, existing_imgs)
    print("\nTraining complete! Run 'tensorboard --logdir lightning_logs/' to view.")

if __name__ == "__main__":
    main()

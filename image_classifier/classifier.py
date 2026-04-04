import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler

DATA_ROOT = "data/humans"
TRAIN_DIR = os.path.join(DATA_ROOT, "training")
TEST_DIR = os.path.join(DATA_ROOT, "test")

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset directory '{TRAIN_DIR}' not found.")
    print("Please ensure the 'humans' folder is extracted in the script's directory.")
    sys.exit(1)

data_transforms = transforms.Compose([
    transforms.Resize(tuple((256, 256))),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=data_transforms)

print(f"Classes found: {train_dataset.classes}")
print(f"Total training images available: {len(train_dataset)}")

rnd_sampler = RandomSampler(
    train_dataset, 
    num_samples=200, # Only draw 200 samples per loop
    replacement=True  # Required when num_samples is used
)

train_loader = DataLoader(train_dataset, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=True)

# TODO: add comments
# implement AMP and early stopping
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
        x = self.features(x)
        x = self.classify(x)
        
        if probability_flag:
            return F.softmax(x, dim=1)
        
        return x

def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer):
    model.train()
    print(f"\n--- Epoch {epoch + 1} (Sampling {len(dataloader.sampler)} images) ---")
    
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 10 == 0:
            print(f"  Batch {batch}: Loss = {loss.item():>7f}")
        if batch >= 200: break
        
        writer.add_scalar('Loss/Train', loss.item(), epoch) # .item() needed because loss is calc'd as tensor, not float


def evaluate(dataloader, model, loss_fn, epoch, writer):
    model.eval()
    test_loss, correct, total= 0, 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            # calculate loss
            pred = model(X)
            test_loss = loss_fn(pred, y).item()

            # accuracy 
            prob = model(X, probability_flag=True)
            confidence, predicted_class = torch.max(prob, dim=1)
            
            correct += (predicted_class == y).sum().item()
            total += y.size(0)
            
    accuracy = (correct / total) * 100
    writer.add_scalar('Loss/Validation', test_loss, epoch)
            
    print(f"  Evaluation: Accuracy = {accuracy:>0.1f}%")
    print(f"  Epoch {epoch + 1}: eval_Loss = {test_loss:>7f}")
    # early stop
    

def main():
    model = HumanIdentificationModel()
    print(model)
    NUM_EPOCHS = 2
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir='./image_classifier/runs/metrics_lab') 

    for epoch in range(NUM_EPOCHS):
        train_loop(train_loader, model, criterion, optimizer, epoch, writer)
        evaluate(test_loader, model, criterion, epoch, writer)

    writer.close()
    print("\nTraining complete! Run 'tensorboard --logdir=image_classifier/runs' to view.")

if __name__ == "__main__":
    main()

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms

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

class HumanModel(nn.Module):
    def __init__(self):
        super(HumanModel, self).__init__()
        self.flatten = nn.Flatten()

        self.features = nn.Sequential(
            # nn.Linear(2352, 128),
            # nn.ReLU(),
            # nn.Linear(128, 2)
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

    def forward(self, x):
        x = self.features(x)
        x = self.classify(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    print(f"\n--- Epoch {epoch+1} (Sampling {len(dataloader.sampler)} images) ---")
    
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 2 == 0:
            print(f"  Batch {batch}: Loss = {loss.item():>7f}")
        if batch >= 500: break

def evaluate(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct, total= 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            total += 1
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # We only evaluate the first batch for speed in this demo
            break
            
    print(f"  Evaluation: Accuracy = {100 * correct / total:>0.1f}%")

def main():
    model = HumanModel()
    print(model)
    NUM_EPOCHS = 2
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train_loop(train_loader, model, criterion, optimizer, epoch)
        evaluate(test_loader, model, criterion)

if __name__ == "__main__":
    main()
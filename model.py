import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import pytz

class LightweightCNN(nn.Module):
    def __init__(self):
        super(LightweightCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        
        self.conv2 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 20, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(20)
        
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.LeakyReLU(0.1)
        
        self.fc1 = nn.Linear(20 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.dropout(x)
        x = x.view(-1, 20 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_ist_time():
    """Get current time in IST format"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')

def train_epoch(model, train_loader, criterion, optimizer, device, epoch=1):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    num_batches = len(train_loader)
    
    print(f"[{get_ist_time()}] Starting Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Log every 100 batches
        if (batch_idx + 1) % 100 == 0:
            current_accuracy = 100 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            print(f"[{get_ist_time()}] Epoch {epoch}, Batch {batch_idx + 1}/{num_batches}, "
                  f"Loss: {avg_loss:.4f}, Accuracy: {current_accuracy:.2f}%")
    
    final_accuracy = 100 * correct / total
    avg_loss = total_loss / num_batches
    print(f"[{get_ist_time()}] Epoch {epoch} completed - "
          f"Final Loss: {avg_loss:.4f}, Final Accuracy: {final_accuracy:.2f}%")
    
    return final_accuracy

class AugmentedMNIST(Dataset):
    def __init__(self, root='./data', train=True):
        self.original_dataset = datasets.MNIST(root=root, train=train, download=True)
        
        # More subtle augmentation parameters
        self.augmentation = transforms.Compose([
            transforms.RandomAffine(
                degrees=5,  # Reduced rotation range from 10 to 5 degrees
                translate=(0.05, 0.05),  # Reduced translation from 0.1 to 0.05
                scale=(0.95, 1.05),  # Reduced scale range from (0.9, 1.1) to (0.95, 1.05)
                fill=0
            ),
            transforms.ToTensor(),
        ])
        
        self.basic_transform = transforms.ToTensor()

    def __getitem__(self, index):
        img, label = self.original_dataset[index]
        
        # Reduced probability from 30% to 20%
        if torch.rand(1) < 0.2:
            img = self.augmentation(img)
        else:
            img = self.basic_transform(img)
            
        return img, label

    def __len__(self):
        return len(self.original_dataset)

def load_data(use_augmentation=False):
    """
    Load MNIST data with optional augmentation
    Args:
        use_augmentation (bool): If True, use augmented dataset
    """
    if use_augmentation:
        train_dataset = AugmentedMNIST('./data', train=True)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    return train_loader

if __name__ == "__main__":
    print(f"[{get_ist_time()}] Starting MNIST Training")
    
    device = torch.device("cpu")
    model = LightweightCNN().to(device)
    
    param_count = count_parameters(model)
    print(f"[{get_ist_time()}] Model initialized with {param_count:,} parameters")
    
    train_loader = load_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    print(f"[{get_ist_time()}] Training configuration:")
    print(f"  - Device: {device}")
    print(f"  - Optimizer: Adam (lr=0.003)")
    print(f"  - Loss: CrossEntropyLoss")
    print(f"  - Batch size: 32")
    print(f"  - Total batches per epoch: {len(train_loader)}")
    
    torch.manual_seed(42)
    accuracy = train_epoch(model, train_loader, criterion, optimizer, device, epoch=1)
    
    print(f"[{get_ist_time()}] Training completed!")
    print(f"[{get_ist_time()}] Final first epoch accuracy: {accuracy:.2f}%")
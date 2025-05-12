from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config
from pathlib import Path
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import numpy as np

# Define image transformations
transform_test_val = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])



# Define transform_train
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Set up paths
data_dir = config.datapath / "chest_xray"
train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"

# Create datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_test_val)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test_val)

#Create sampler to equalize class distribution
class_counts = Counter([label for _, label in train_dataset.samples])
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

unique, counts = np.unique(train_dataset.targets, return_counts=True)
print(f"New class distribution: {dict(zip(unique, counts))}")
